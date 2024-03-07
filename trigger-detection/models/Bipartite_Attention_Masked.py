from os import X_OK
from sklearn import utils
from torch import tensor
from torch._C import is_anomaly_enabled
import torch.nn as nn
import torch
from typing import OrderedDict, Tuple
from icecream import ic
from torch.nn.modules import distance
from torch_scatter import scatter_add

from random import sample
import logging
from icecream import ic
import sys

# local
from .utils_settrans import Masked_SAB
from .layers import Masked_LayerNorm
from .Bipartite_Attention_gLRI import get_approximate_radii, get_track_endpoints, get_predicted_pz

class Bipartite_Attention(nn.Module):
    def __init__(
            self,
            num_features,
            num_classes,
            layers_spec, # List of tuple of ((feature_dim, n_aggregators))
            hidden_activation='Tanh', 
            aggregator_activation='softmax',
            ln=False,
            bn=False,
            recalculate_hits_mean=True,
            self_split=False,
            final_pooling=True
            ):
        super().__init__()
        self.final_pooling = final_pooling
        self.recalculate_hits_mean = recalculate_hits_mean
        self.self_split = self_split
        self.bn = bn

        _layers = []
        prev_dim = num_features
        for feature_dim, n_aggregators in layers_spec:
            _layers.append(
                    Bipartite_Layers(
                        prev_dim,
                        feature_dim,
                        n_aggregators,
                        hidden_activation,
                        aggregator_activation,
                        ln=ln
                    )
            )
            prev_dim = feature_dim 

        self._layers = nn.ModuleList(_layers)

        if self.final_pooling:
            self.score = nn.Sequential(
                    nn.Linear(prev_dim, 1)
            )

            if bn:
                self._pred_layers = nn.ModuleList([
                        nn.Linear((4 if self.self_split else 1)*2*prev_dim, 2*prev_dim),
                        getattr(nn, hidden_activation)(),
                        nn.BatchNorm1d(2*prev_dim),
                        nn.Linear(2*prev_dim, num_classes)
                ])

                def _pred(X, mask):
                    X = self._pred_layers[0](X)
                    X = self._pred_layers[1](X)
                    X = self._pred_layers[2](X)
                    X = self._pred_layers[3](X)
                    return X

                self._pred = _pred
            else:
                self._pred_layers = nn.ModuleList([
                        nn.Linear((4 if self.self_split else 1)*2*prev_dim, 2*prev_dim),
                        getattr(nn, hidden_activation)(),
                        nn.Linear(2*prev_dim, num_classes)
                ])

                def _pred(X, mask):
                    X = self._pred_layers[0](X)
                    X = self._pred_layers[1](X)
                    X = self._pred_layers[2](X)
                    return X

                self._pred = _pred
        else:
            if ln:
                self.convert = nn.Sequential(
                        nn.Linear(prev_dim, 2*prev_dim),
                        # It's ok as we immediately multiply the results by the mask
                        nn.LayerNorm(2*prev_dim),
                        getattr(nn, hidden_activation)()
                )
                self._pred_layers = nn.ModuleList([
                        nn.Linear((4 if self.self_split else 1)*2*prev_dim, 2*prev_dim),
                        Masked_LayerNorm(2*prev_dim),
                        getattr(nn, hidden_activation)(),
                        nn.Linear(2*prev_dim, num_classes)
                ])

                def _pred(X, mask):
                    X = self._pred_layers[0](X)
                    X = self._pred_layers[1](X, mask)
                    X = self._pred_layers[2](X)
                    X = self._pred_layers[3](X)
                    return X * mask.unsqueeze(-1)

                self._pred = _pred

            else:
                self.convert = nn.Sequential(
                        nn.Linear(prev_dim, 2*prev_dim),
                        # It's ok as we immediately multiply the results by the mask
                        getattr(nn, hidden_activation)()

                )
                self._pred_layers = nn.ModuleList([
                        nn.Linear((4 if self.self_split else 1)*2*prev_dim, 2*prev_dim),
                        getattr(nn, hidden_activation)(),
                        nn.Linear(2*prev_dim, num_classes)
                ])

                def _pred(X, mask):
                    X = self._pred_layers[0](X)
                    X = self._pred_layers[1](X)
                    X = self._pred_layers[2](X)
                    return X * mask.unsqueeze(-1)
                self._pred = _pred

        if ln: 
            self._track_project = nn.Sequential(
                    nn.Linear(prev_dim, 2*prev_dim),
                    nn.LayerNorm(2*prev_dim),
                    getattr(nn, hidden_activation)(),
                    nn.Linear(2*prev_dim, prev_dim)
            )
        else:
            self._track_project = nn.Sequential(
                    nn.Linear(prev_dim, 2*prev_dim),
                    getattr(nn, hidden_activation)(),
                    nn.Linear(2*prev_dim, prev_dim)
            )



    def forward(self, X, mask):
        if self.self_split:
            hits = X[:, :, :15] * mask.unsqueeze(-1)
            hits = hits.reshape(X.shape[0], X.shape[1], 5, 3)

            hits_mask = hits[:, :, :, 0] >= 0
            Xp, maskp = self._mask_filter(X, mask, hits, hits_mask)
            Xp = self._recalculate_hits_mean(Xp, maskp)
            h1 = self._generate_track_embeddings(Xp, maskp)
            h1 = self._pool_track_embeddings(h1, maskp) if self.final_pooling else h1

            hits_mask = hits[:, :, :, 0] < 0
            Xp, maskp = self._mask_filter(X, mask, hits, hits_mask)
            Xp = self._recalculate_hits_mean(Xp, maskp)
            h2 = self._generate_track_embeddings(Xp, maskp)
            h2 = self._pool_track_embeddings(h2, maskp) if self.final_pooling else h2

            hits_mask = hits[:, :, :, 1] >= 0
            Xp, maskp = self._mask_filter(X, mask, hits, hits_mask)
            Xp = self._recalculate_hits_mean(Xp, maskp)
            h3 = self._generate_track_embeddings(Xp, maskp)
            h3 = self._pool_track_embeddings(h3, maskp) if self.final_pooling else h3

            hits_mask = hits[:, :, :, 1] < 0
            Xp, maskp = self._mask_filter(X, mask, hits, hits_mask)
            Xp = self._recalculate_hits_mean(Xp, maskp)
            h4 = self._generate_track_embeddings(Xp, mask)
            h4 = self._pool_track_embeddings(h4, maskp) if self.final_pooling else h4

            # This only loosely makes sense
            # Track embeddings generated for a split in which the track does not lie
            # will be 0.
            H = torch.cat([h1, h2, h3, h4], dim=-1)
        else:
            if self.recalculate_hits_mean:
                Xp = self._recalculate_hits_mean(X, mask)
            else:
                Xp = X
            H = self._generate_track_embeddings(Xp, mask)

        H = self._pool_track_embeddings(H, mask) if self.final_pooling else self.convert(H)*mask.unsqueeze(-1)

        return self._pred(H, mask)

    def generate_track_embeddings(self, X, mask):
        if self.self_split:
            raise NotImplementedError("Splitting not implemented for generating track embeddings")

        if self.recalculate_hits_mean:
                X = self._recalculate_hits_mean(X, mask)

        H_t = self._generate_track_embeddings(X, mask) * mask.unsqueeze(-1)

        if self.final_pooling:
            H_e = self._pool_track_embeddings(H_t, mask)
        else:
            H_e = self.convert(H_t)*mask.unsqueeze(-1)

        return H_t, self._pred(H_e, mask)

    def predict_adjacency_matrix(self, H_t, mask):
        H_t = self._track_project(H_t)
        # H_t: (batch, track, n)
        A = torch.sum(H_t.unsqueeze(-2) * H_t.unsqueeze(-3), dim=-1) 
        A = A * (mask.unsqueeze(-1) * mask.unsqueeze(-2))
        return A

    @staticmethod
    def recalculate_geometric_features(X, e_v, **config):
        hits = X[..., :15].reshape(X.shape[0], X.shape[1], 5, 3)
        good_hits = torch.any(hits != 0, dim=-1)
        # (batch, track, 5)
        good_hits_mask = good_hits
        good_hits = good_hits.unsqueeze(-1).repeat(1, 1, 1, 3).reshape(X.shape[0], X.shape[1], 15)

        e_v = e_v.reshape(X.shape[0], X.shape[1], 15)
        Xp = (X[..., :15] + e_v)*good_hits

        hits = Xp[..., :15].reshape(X.shape[0], X.shape[1], 5, 3)
        if config['add_geo_features']:
            geo_features = torch.zeros(X.shape[0], X.shape[1], 13).to(X.device)
            #phi = torch.zeros((X.shape[0], X.shape[1], 5)).to(X.device)
            for i in range(4):
                geo_features[:, :, i] = torch.linalg.norm(hits[:, :, i + 1] - hits[:, :, i], ord=2, dim=(-1,))

            geo_features[:, :, 5] = torch.linalg.norm(hits[:, :, 4] - hits[:, :, 0], ord=2, dim=(-1,))
            x_hits = hits[..., 0] + hits[..., 0] == 0
            phi = torch.atan2(hits[..., 1].to(torch.float64), x_hits.to(torch.float64))
            geo_features[:, :, 6:10] = torch.diff(phi, dim=-1).to(torch.float32)

            Xp = torch.cat([Xp, geo_features], dim=-1)

        r = None
        centers = None
        n_hits = None
        if config['use_radius']:
            n_hits = torch.sum(good_hits_mask, dim=-1)
            r, centers = get_approximate_radii(Xp[:, :, :15], n_hits, good_hits_mask, False)

            Xp = torch.cat([Xp, r.unsqueeze(-1)], dim=-1)
        if config['use_center']:
            if n_hits is None:
                n_hits = torch.sum(good_hits_mask, dim=-1)
            if centers is None:
                r, centers = get_approximate_radii(Xp[:, :, :15], n_hits, good_hits_mask, False)
            Xp = torch.cat([Xp, centers], dim=-1)


        if config['use_predicted_pz']:
            if r is None:
                n_hits = torch.sum(good_hits_mask, dim=-1)
                r, centers = get_approximate_radii(Xp[:, :, :15], n_hits, good_hits_mask, False)

            first_hits, last_hits = get_track_endpoints(hits, good_hits_mask)
            pred_pz = get_predicted_pz(first_hits, last_hits, r)
            Xp = torch.cat([Xp, pred_pz.unsqueeze(-1)], dim=-1)
        
        if config['use_n_pixels']:
            if config['use_n_hits']:
                Xp = torch.cat([Xp, X[..., -10:]], dim=-1)
            else:
                Xp = torch.cat([Xp, X[..., -5:]], dim=-1)
        elif config['use_n_hits']:
            Xp = torch.cat([Xp, X[..., -5:]], dim=-1)

        return Xp, good_hits_mask


    def generate_embedding(self, X, mask):
        if self.self_split:
            hits = X[:, :, :15] * mask.unsqueeze(-1)
            hits = hits.reshape(X.shape[0], X.shape[1], 5, 3)

            hits_mask = hits[:, :, :, 0] >= 0
            Xp, maskp = self._mask_filter(X, mask, hits, hits_mask)
            Xp = self._recalculate_hits_mean(Xp, maskp)
            h1 = self._generate_track_embeddings(Xp, maskp)
            if self.final_pooling:
                h1 = self._pool_track_embeddings(h1, maskp)

            hits_mask = hits[:, :, :, 0] < 0
            Xp, maskp = self._mask_filter(X, mask, hits, hits_mask)
            Xp = self._recalculate_hits_mean(Xp, maskp)
            h2 = self._generate_track_embeddings(Xp, maskp)
            if self.final_pooling:
                h2 = self._pool_track_embeddings(h2, maskp)

            hits_mask = hits[:, :, :, 1] >= 0
            Xp, maskp = self._mask_filter(X, mask, hits, hits_mask)
            Xp = self._recalculate_hits_mean(Xp, maskp)
            h3 = self._generate_track_embeddings(Xp, maskp)
            if self.final_pooling:
                h3 = self._pool_track_embeddings(h3, maskp)

            hits_mask = hits[:, :, :, 1] < 0
            Xp, maskp = self._mask_filter(X, mask, hits, hits_mask)
            Xp = self._recalculate_hits_mean(Xp, maskp)
            h4 = self._generate_track_embeddings(Xp, mask)
            if self.final_pooling:
                h4 = self._pool_track_embeddings(h4, maskp)

            if self.final_pooling:
                H = torch.cat([h1, h2, h3, h4], dim=-1)
            else:
                H = torch.stack([h1, h2, h3, h4], dim=1)
        else:
            if self.recalculate_hits_mean:
                X = self._recalculate_hits_mean(X, mask)
            H = self._generate_track_embeddings(X, mask)
            if self.final_pooling:
                H = self._pool_track_embeddings(H, mask)


        return H

    def _generate_track_embeddings(self, X, mask):
        for layer in self._layers:
            pred_x, agg = layer(X, mask)
            X = pred_x

        return X

    def _pool_track_embeddings(self, X, mask):
        attention_score = self.score(X)
        attention_score = self._masked_softmax(attention_score, mask)
        edges = torch.einsum('btf,bta->baft', X, attention_score)
        min_elem = torch.min(edges, dim=-1)[0] - 1
        edges_premaxpool = edges*mask.reshape(mask.shape[0], 1, 1, mask.shape[1]) + (1 - mask).reshape(mask.shape[0], 1, 1, mask.shape[1])*(min_elem.unsqueeze(-1)) # set all masked edges -torch.inf
        max_pooled = torch.max(edges_premaxpool, dim=-1)[0] # (n_minibatches, n_aggregators, n_features)
        mean_pooled = self._masked_mean(edges, mask)
        H = torch.cat((max_pooled, mean_pooled), axis=-1).squeeze(1)
        return H

    @staticmethod
    def _masked_softmax(x, mask):
        max_elem = torch.max(x, dim=1)[0]
        xp = x - max_elem.unsqueeze(1)
        num = torch.exp(xp)*mask.unsqueeze(-1)
        dem = torch.sum(num, dim=1).unsqueeze(1)
        return num/(dem + 1e-16)

    @staticmethod
    def _masked_mean(x, mask):
        # : (minibatch, aggregator, feature, track)
        xp = x * mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
        summed = torch.sum(xp, dim=-1) # (batch, aggregator, feature)
        # mask: (batch, track
        n_tracks = torch.sum(mask, dim=-1).reshape(summed.shape[0], 1, 1)
        n_tracks[n_tracks == 0] = 1
        return summed / n_tracks

    @staticmethod
    def _mask_filter(X, mask, hits, hits_mask):
        hits = torch.clone(hits)
        hits[~hits_mask] = 0
        track_vector = hits.reshape(X.shape[0], X.shape[1], 15)
        empty_tracks = torch.all(track_vector == 0, dim=-1)
        Xp = torch.clone(X)
        Xp[:, :, :15] = track_vector
        maskp = mask * ~empty_tracks
        Xp *= maskp.unsqueeze(-1)
        return Xp, maskp

    @staticmethod
    def _recalculate_hits_mean(X, mask):
        # this will ensure all masked tracks have 0 in their hits
        # shape (minibatch, track, 15)
        Xp = torch.zeros_like(X)
        Xp[:, :, :] = X[:, :, :]
        hits = X[:, :, :15] * mask.unsqueeze(-1)

        # (minibatch, track, layer, coords)
        hits = hits.reshape((X.shape[0], X.shape[1], 5, 3))

        # (minibatch, coords)
        total = torch.sum(hits, dim=(1, 2))

        # (minibatch, track, layer)
        good_hits = torch.all(hits != 0, dim=-1)

        # (minibatch,)
        n_good_hits = torch.sum(good_hits * mask.unsqueeze(-1), dim=(1, 2))
        n_good_hits[n_good_hits == 0] = 1

        hits_mean = total / n_good_hits.unsqueeze(-1)
        Xp[:, :, (15+10):(15+13)] = hits_mean.unsqueeze(1)
        Xp= Xp* mask.unsqueeze(-1)

        return Xp


class Bipartite_Layers(nn.Module):
    def __init__(self,
        input_dim,
        feature_dim,
        n_aggregators,
        hidden_activation,
        aggregator_activation,
        ln,
        num_heads=4,
    ):
        super(Bipartite_Layers, self).__init__()
        self.aggregator_activation = aggregator_activation
        self.enc = nn.ModuleList([
            Masked_SAB(input_dim, feature_dim, num_heads, ln=ln),
            Masked_SAB(feature_dim, feature_dim, num_heads, ln=ln),
        ])
        self.transform_in = nn.Linear(feature_dim, feature_dim)
        self.aggregator_score = nn.Linear(feature_dim, n_aggregators)
        if ln:
            self.transform_out = nn.Sequential(
                nn.Linear(2*feature_dim*n_aggregators + input_dim + feature_dim, feature_dim),
                # We can use LayerNorm directly because it gets immediately masked out in the code
                # that calls transform_out
                nn.LayerNorm(feature_dim),
                getattr(nn, hidden_activation)()
            )
        else:
            self.transform_out = nn.Sequential(
                nn.Linear(2*feature_dim*n_aggregators + input_dim + feature_dim, feature_dim),
                getattr(nn, hidden_activation)()
            )

        self._feature_dim = feature_dim
        self._n_aggregators = n_aggregators

    def forward(self, X, mask):
        """
        X: Tensor of shape [n_minibatches, n_tracks, n_track_features]
        """
        Xp = self.enc[0](X, mask) * mask.unsqueeze(-1)
        Xp = self.enc[1](Xp, mask) * mask.unsqueeze(-1)
        Xp = self.transform_in(Xp) * mask.unsqueeze(-1)
        attention_score = self.aggregator_score(Xp)

        attention_score = self._masked_softmax(attention_score, mask)
        edges = torch.einsum('btf,bta->baft', Xp, attention_score)
        min_elem = torch.min(edges, dim=-1)[0] - 1
        edges_premaxpool = edges*mask.reshape(mask.shape[0], 1, 1, mask.shape[1]) + (1 - mask).reshape(mask.shape[0], 1, 1, mask.shape[1])*(min_elem.unsqueeze(-1)) # set all masked edges -torch.inf
        max_pooled = torch.max(edges_premaxpool, dim=-1)[0] # (n_minibatches, n_aggregators, n_features)
        mean_pooled = self._masked_mean(edges, mask)
        aggregator = torch.cat((max_pooled, mean_pooled), axis=-1)
        aggregator = aggregator.reshape(aggregator.shape[0], aggregator.shape[1]*aggregator.shape[2])
        agg_repeated = aggregator.reshape(aggregator.shape[0], 1, -1).repeat(1, X.shape[1], 1)
        H = torch.cat((X, Xp, agg_repeated), axis=-1)

        return self.transform_out(H)*mask.unsqueeze(-1), aggregator

    @staticmethod
    def _masked_softmax(x, mask):
        max_elem = torch.max(x, dim=1)[0]
        xp = x - max_elem.unsqueeze(1)
        num = torch.exp(xp)*mask.unsqueeze(-1)
        dem = torch.sum(num, dim=1).unsqueeze(1)
        return num/(dem + 1e-16)

    @staticmethod
    def _masked_mean(x, mask):
        # : (minibatch, aggregator, feature, track)
        xp = x * mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
        summed = torch.sum(xp, dim=-1) # (batch, aggregator, feature)
        # mask: (batch, track
        n_tracks = torch.sum(mask, dim=-1).reshape(summed.shape[0], 1, 1)
        n_tracks[n_tracks == 0] = 1
        return summed / n_tracks
