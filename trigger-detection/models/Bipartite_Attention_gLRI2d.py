from os import X_OK
from sklearn import utils
from torch import tensor
from torch._C import is_anomaly_enabled
import torch.nn as nn
import torch
from typing import OrderedDict, Tuple
from icecream import ic
from torch.nn.modules import distance

from random import sample
import logging
from icecream import ic
import sys
import torch.nn.functional as F
import numpy as np

# local
from .utils_settrans import Masked_SAB

def matmul_3D(A, B):
    return torch.einsum('...ij,...jk->...ik', A, B)

def get_approximate_radii(tracks_info, n_hits, good_hits, use_log):
    x_indices = [3*j for j in range(5)]
    y_indices = [3*j+1 for j in range(5)]
    r = torch.zeros((*tracks_info.shape[:2], 1), device=tracks_info.device, dtype=torch.float64)

    centers = torch.zeros((*tracks_info.shape[:2], 2), device=tracks_info.device, dtype=torch.float64)
    for n_hit in range(3, 5 + 1):
        complete_tracks = tracks_info[n_hits == n_hit]
        hit_indices = good_hits[n_hits == n_hit]
        if complete_tracks.shape[0] == 0:
            continue

        A = torch.ones((complete_tracks.shape[0], n_hit, 3), device=tracks_info.device, dtype=torch.float64)
        x_values = complete_tracks[..., x_indices]
        x_values = x_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)

        y_values = complete_tracks[..., y_indices].to(torch.float64)
        y_values = y_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)
        A[..., 0] = x_values
        A[..., 1] = y_values

        y = - x_values**2 - y_values**2
        y = y.unsqueeze(-1)
        AT = A.transpose(-1, -2)
        c = matmul_3D(matmul_3D(torch.linalg.inv(matmul_3D(AT, A)), AT), y)

        r[n_hits == n_hit] = torch.sqrt(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2])/200
        centers[n_hits == n_hit] = torch.cat([-c[:, 0]/2, -c[:, 1]/2], dim=-1)
        if use_log:
            r[n_hits == n_hit] = torch.log(r[n_hits == n_hit])

    return r.squeeze(-1).to(torch.float32), centers.to(torch.float32)

def get_track_endpoints(hits, good_hits):
    # Assumption: all tracks have at least 1 hit
    # If it has one hit, first_hit == last_hit for that track
    # hits shape: (n_tracks, 5, 3)
    # good_hits shape: (n_tracks, 5)
    d = hits.device
    min_indices = good_hits * torch.arange(5, device=d) + ~good_hits * torch.arange(5, 10, device=d)
    indices = torch.argmin(min_indices, dim=-1).unsqueeze(-1).unsqueeze(-2)
    first_hits = torch.take_along_dim(hits, indices, dim=-2)
    max_indices = good_hits * torch.arange(5, 10, device=d) + ~good_hits * torch.arange(5, device=d)
    indices = torch.argmax(max_indices, dim=-1).unsqueeze(-1).unsqueeze(-2)
    last_hits = torch.take_along_dim(hits, indices, dim=-2)

    return first_hits.squeeze(-2), last_hits.squeeze(-2)

def get_predicted_pz(first_hit, last_hit, radius):
    dz = (last_hit[..., -1] - first_hit[..., -1])/100
    chord2 = ((last_hit[..., 0] - first_hit[..., 0]) ** 2 + (last_hit[..., 1] - first_hit[..., 1]) ** 2) / 10000
    inner = (2*radius**2 - chord2) / (2*radius**2 + 1e-8)
    inner = (inner <= -1 + 1e-6) * (-1 + 1e-6) + ((-1 + 1e-6 < inner) & (inner < 1 - 1e-6)) * inner + (inner >= 1 - 1e-6)*(1 - 1e-6)
	
    dtheta = torch.acos(inner)
    theta = (dtheta == 0) * 1 + dtheta
    return dz/dtheta




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
            final_pooling=True,
            add_geo_features=True,
            use_predicted_pz=True,
            use_radius=True,
            use_center=True,
            use_log_recalculated_radius=False,
            use_cholesky=False
            ):
        super().__init__()
        from .Bipartite_Attention_Masked import Bipartite_Attention as BA
        self.use_cholesky = use_cholesky
        self.add_geo_features = add_geo_features
        self.use_predicted_pz = use_predicted_pz
        self.use_radius = use_radius
        self.use_center = use_center
        self.interpreter = BA(
                num_features=num_features, 
                num_classes=num_classes, 
                layers_spec=layers_spec, 
                hidden_activation=hidden_activation,
                aggregator_activation=aggregator_activation,
                ln=ln,
                bn=bn,
                recalculate_hits_mean=recalculate_hits_mean,
                self_split=False,
                final_pooling=False
        )
        self.use_log = use_log_recalculated_radius


        act = getattr(nn, hidden_activation)
        self.self_split = self_split
        if self.self_split:
            assert final_pooling, "We have not implement final_pooling=False when splitting"
            self.pooler = BA(
                    num_features=layers_spec[-1][0],
                    num_classes=layers_spec[-1][0], 
                    layers_spec=layers_spec, 
                    hidden_activation=hidden_activation,
                    aggregator_activation=aggregator_activation,
                    ln=ln,
                    bn=bn,
                    recalculate_hits_mean=True,
                    self_split=False,
                    final_pooling=True
            )
            self.classifier = nn.Sequential(
                    nn.Linear(layers_spec[-1][0]*2, layers_spec[-1][0]),
                    nn.LayerNorm(layers_spec[-1][0]),
                    act(),
                    nn.Linear(layers_spec[-1][0], layers_spec[-1][0]),
                    nn.LayerNorm(layers_spec[-1][0]),
                    act(),
                    nn.Linear(layers_spec[-1][0], num_classes),
                )
        else:
            self.classifier = BA(
                    num_features=num_features,
                    num_classes=num_classes, 
                    layers_spec=layers_spec, 
                    hidden_activation=hidden_activation,
                    aggregator_activation=aggregator_activation,
                    ln=ln,
                    bn=bn,
                    recalculate_hits_mean=False,
                    self_split=False,
                    final_pooling=final_pooling
            )


        if ln:
            self.noiser = nn.Sequential(
                    nn.Linear(layers_spec[-1][0], layers_spec[-1][0]),
                    nn.LayerNorm(layers_spec[-1][0]),
                    act(),
                    nn.Linear(layers_spec[-1][0], layers_spec[-1][0]),
                    nn.LayerNorm(layers_spec[-1][0]),
                    act(),
                    nn.Linear(layers_spec[-1][0], 5*(2*2 + 2)),
                )
        else:
            self.noiser = nn.Sequential(
                    nn.Linear(layers_spec[-1][0], 64),
                    act(),
                    nn.Linear(64, 64),
                    act(),
                    nn.Linear(layers_spec[-1][0], 5*(2*2+2))
                )



    def forward(self, X, mask, detach_radius_grad=False, finetune_epoch=False):
        if self.self_split:
            raise NotImplementedError("self_split not implemented.")
        else:
            if finetune_epoch:
                track_embeddings = self.interpreter.generate_embedding(X, mask)

                noise = self.noiser(track_embeddings) * mask.unsqueeze(-1)
                U = noise[..., :5*4].reshape(X.shape[0], X.shape[1], 5, 2, 2)
                a = F.softplus(noise[..., 5*4:]).reshape(X.shape[0], X.shape[1], 5, 2)
                a = torch.clamp(a, 1e-6, 1e6)
                I = torch.eye(2, device=X.device).reshape(1, 1, 1, 2, 2)
                sigma_v = a[..., 0:1].unsqueeze(-1)*torch.einsum('bthij,bthjk->bthik', U, U.transpose(-1, -2)) + a[..., 1:2].unsqueeze(-1)*I
                if self.use_cholesky:
                    s = torch.normal(
                        torch.zeros((X.shape[0], X.shape[1], 5, 2)).to(X.device),
                        torch.ones((X.shape[0], X.shape[1], 5, 2)).to(X.device)
                    )
                    U = torch.linalg.cholesky(sigma_v)
                    e_v = torch.einsum('bthij,bthj->bthi', U, s)
                else:
                    s = torch.normal(
                            torch.zeros((X.shape[0], X.shape[1], 5, 2, 2)).to(X.device),
                            torch.ones((X.shape[0], X.shape[1], 5, 2, 2)).to(X.device)
                    )
                    # (sigma_v shape: (batch, n_tracks, 5, 2, 2)

                    
                    e_v = torch.sqrt(a[..., 0:1])*torch.einsum('bthij,bthj->bthi', U, s[..., 0]) + \
                            torch.sqrt(a[..., 1:2])*torch.einsum('bthij,bthj->bthi', I, s[..., 1])

                z_noise = torch.zeros(e_v.shape[0], e_v.shape[1], 5, 1, device=X.device)
                e_v = torch.cat([e_v, z_noise], dim=-1)
                e_v = e_v.reshape(X.shape[0], X.shape[1], 15)

                Xp, good_hits_mask = self.recalculate_geometric_features(X, e_v, detach_radius_grad=detach_radius_grad)
                Xp = Xp * mask.unsqueeze(-1)

                pred = self.classifier(Xp, mask)
                return pred, sigma_v, good_hits_mask
            else:
                pred = self.classifier(X, mask)
                return pred, None, None

    def forward_with_e_v(self, X, mask, e_v):
            Xp, good_hits_mask = self.recalculate_geometric_features(X, e_v, detach_radius_grad=False, force=True)
            Xp = Xp * mask.unsqueeze(-1)

            pred = self.classifier(Xp, mask)
            return pred



    def recalculate_geometric_features(self, X, e_v, detach_radius_grad=False, force=False):
        if self.training or force:
            hits = X[..., :15].reshape(X.shape[0], X.shape[1], 5, 3)
            good_hits = torch.any(hits != 0, dim=-1)
            # (batch, track, 5)
            good_hits_mask = good_hits
            good_hits = good_hits.unsqueeze(-1).repeat(1, 1, 1, 3).reshape(X.shape[0], X.shape[1], 15)

            Xp = (X[..., :15] + e_v)*good_hits
            hits = Xp[..., :15].reshape(X.shape[0], X.shape[1], 5, 3)
            if self.add_geo_features:
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
            if self.use_radius:
                n_hits = torch.sum(good_hits_mask, dim=-1)
                r, centers = get_approximate_radii(Xp[:, :, :15], n_hits, good_hits_mask, self.use_log)
                if detach_radius_grad:
                    r = r.detach()

                Xp = torch.cat([Xp, r.unsqueeze(-1)], dim=-1)
            if self.use_center:
                if n_hits is None:
                    n_hits = torch.sum(good_hits_mask, dim=-1)
                if centers is None:
                    r, centers = get_approximate_radii(Xp[:, :, :15], n_hits, good_hits_mask, self.use_log)
                Xp = torch.cat([Xp, centers], dim=-1)


            if self.use_predicted_pz:
                if r is None:
                    n_hits = torch.sum(good_hits_mask, dim=-1)
                    r, centers = get_approximate_radii(Xp[:, :, :15], n_hits, good_hits_mask, False)
                    if detach_radius_grad:
                        r = r.detach()

                first_hits, last_hits = get_track_endpoints(hits, good_hits_mask)
                pred_pz = get_predicted_pz(first_hits, last_hits, r)
                Xp = torch.cat([Xp, pred_pz.unsqueeze(-1)], dim=-1)

            return Xp, good_hits_mask
        else:
            hits = X[..., :15].reshape(X.shape[0], X.shape[1], 5, 3)
            good_hits = torch.any(hits != 0, dim=-1)
            # (batch, track, 5)
            good_hits_mask = good_hits
            good_hits = good_hits.unsqueeze(-1).repeat(1, 1, 1, 3).reshape(X.shape[0], X.shape[1], 15)

            return X, good_hits_mask
