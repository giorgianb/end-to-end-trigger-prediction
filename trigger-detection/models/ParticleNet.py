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


class EdgeConv(nn.Module):
    def __init__(
            self,
            num_features,
            k,
            hidden_dim = 64,
            n_hidden_layers = 3,
            hidden_activation='PReLU',
            ln=False,
            distance='euclidean'
    ):
        super().__init__()
        self.k = k
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.n_hidden_layers = n_hidden_layers
        self.distance = distance

        layers = [nn.Linear(2*num_features, hidden_dim), nn.LayerNorm(hidden_dim),  getattr(nn, hidden_activation)()]
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(getattr(nn, hidden_activation)())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(getattr(nn, hidden_activation)())

        self.layers = nn.Sequential(*layers)


    def forward(self, points, x, mask):
        # x: (batch_size, n_tracks, n_features)
        # mask: (batch_size, n_tracks)
        if self.distance == 'euclidean':
            delta = points.unsqueeze(2) - points.unsqueeze(1) # (batch_size, n_tracks, n_tracks, n_features)
            distances = torch.norm(delta, dim=-1) # (batch_size, n_tracks, n_tracks)
            distances = distances.masked_fill(~mask.to(torch.bool).unsqueeze(1), float('inf'))
        elif self.distance == 'emd':
            norms = torch.sum(points, dim=-1)
            norms[norms == 0] = 1
            pdf = points / norms.unsqueeze(-1)
            cdf = torch.cumsum(pdf, dim=-1)
            total = cdf[:, :, -1]
            # First term moves dirth in between
            # Second term equalizes the total amount of dirt
            distances = torch.sum(torch.abs(cdf.unsqueeze(2) - cdf.unsqueeze(1)), dim=-1) + torch.abs(total.unsqueeze(2) - total.unsqueeze(1))
        k = min(self.k, x.shape[1])
        _, indices = torch.topk(distances, k=k, dim=-1, largest=False) # (batch_size, n_tracks, k)
        n_tracks = torch.sum(mask, dim=-1)
        neighbors = torch.gather(x.unsqueeze(2).repeat(1, 1, k, 1), 1, indices.unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])) # (batch_size, n_tracks, k, n_features)
        h = torch.cat([x.unsqueeze(2).repeat(1, 1, k, 1), x.unsqueeze(2) - neighbors], dim=-1) # (batch_size, n_tracks, k, 2*n_features)
        conv = self.layers(h) # (batch_size, n_tracks, k, n_features)
        neighbors_mask = torch.arange(k).to(x.device).unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1) < n_tracks.unsqueeze(-1).unsqueeze(-1)
        n_tracks[n_tracks == 0] = 1
        aggregated = torch.sum(conv * neighbors_mask.unsqueeze(-1).to(torch.float), dim=-2) / n_tracks.unsqueeze(-1).unsqueeze(-1).to(torch.float) # (batch_size, n_tracks, n_features)
        if torch.any(torch.isnan(aggregated)):
            ic(torch.any(torch.isnan(aggregated)))
            ic(torch.any(torch.isnan(conv)))
            ic(torch.any(torch.isnan(neighbors_mask)))
            ic(torch.any(torch.isnan(h)))
            ic(torch.any(torch.isnan(neighbors)))
            ic(torch.any(torch.isnan(x)))
            sys.exit()
        return aggregated
        
class ParticleNet(nn.Module):
    def __init__(
            self,
            num_features,
            num_classes,
            layers_spec, # List of tuple of ((hidden_dim, n_layers, n_neighbors))
            hidden_activation='PReLU', 
            ln=False,
            bn=False,
            recalculate_hits_mean=True,
            distance='euclidean'
            ):
        super().__init__()
        self.recalculate_hits_mean = recalculate_hits_mean
        self.bn = bn

        _layers = []
        prev_dim = num_features
        for feature_dim, n_layers, k in layers_spec:
            _layers.append(
                    EdgeConv(
                        prev_dim,
                        k,
                        feature_dim,
                        n_layers,
                        hidden_activation,
                        ln=ln,
                        distance=distance
                    )
                )
            prev_dim = feature_dim 

        self._layers = nn.ModuleList(_layers)
        self._pred_layers = nn.Sequential(
                nn.Linear(2*prev_dim, 2*prev_dim),
                getattr(nn, hidden_activation)(),
                nn.BatchNorm1d(2*prev_dim),
                nn.Linear(2*prev_dim, num_classes)
        )



    def forward(self, X, mask):
        if self.recalculate_hits_mean:
            X = self._recalculate_hits_mean(X, mask)

        H = self._generate_track_embeddings(X, mask)
        H = self._pool_track_embeddings(H, mask)

        return self._pred_layers(H)

    def _generate_track_embeddings(self, X, mask):
        X = X * mask.unsqueeze(-1)
        points = X[:, :, :15]
        for layer in self._layers:
            pred_x = layer(points, X, mask)
            X = pred_x
            points = pred_x

        return X

    def _pool_track_embeddings(self, X, mask):
        h1 = self._masked_mean(X, mask)
        h2 = self._masked_max(X, mask)

        H = torch.cat([h1, h2], dim=-1)
        
        return H

    @staticmethod
    def _masked_mean(x, mask):
        # : (minibatch, track, feature)
        summed = torch.sum(x * mask.unsqueeze(-1), dim=1)
        n_tracks = torch.sum(mask, dim=-1).reshape(summed.shape[0], 1)
        n_tracks[n_tracks == 0] = 1
        return summed / n_tracks

    @staticmethod
    def _masked_max(x, mask):
        # x: (minibatch, track, feature)
        x = x * mask.unsqueeze(-1) + (1 - mask.unsqueeze(-1)) * torch.min(x, dim=1)[0].unsqueeze(1)
        ret = torch.max(x, dim=1)[0]
        return ret

    @staticmethod
    def _recalculate_hits_mean(X, mask):
        # this will ensure all masked tracks have 0 in their hits
        # shape (minibatch, track, 15)
        hits = X[:, :, :15] * mask.unsqueeze(-1)

        # (minibatch, track, layer, coords)
        hits = hits.reshape((X.shape[0], X.shape[1], 5, 3))

        # (minibatch, coords)
        total = torch.sum(hits, dim=(1, 2))

        # (minibatch, track, layer)
        good_hits = torch.all(hits != 0, dim=-1)

        # (minibatch,)
        n_good_hits = torch.sum(good_hits, dim=(1, 2))
        n_good_hits[n_good_hits == 0] = 1

        hits_mean = total / n_good_hits.unsqueeze(-1)
        X[:, :, (15+10):(15+13)] = hits_mean.unsqueeze(1)
        X = X * mask.unsqueeze(-1)

        return X

