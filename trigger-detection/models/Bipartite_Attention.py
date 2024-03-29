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
# from utils import isParallelMatrix, shortest_dist_parallel_matrix, shortest_dist_non_parallel_matrix

from random import sample
import logging
from icecream import ic
import sys

# local
from .utils_settrans import ISAB, PMA, SAB

class Bipartite_Attention4(nn.Module):
    def __init__(
            self,
            num_features,
            layers_spec, # Tuple of (N, feature_dim, coordinate_dim)
            num_classes,
            hidden_activation='Tanh', 
            aggregator_activation='potential',
            ln=False,
            ):
        super(Bipartite_Attention4, self).__init__()
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
                        ln=ln,
                    )
            )
            prev_dim = feature_dim 

        self._layers = nn.ModuleList(_layers)

        # self._layers = nn.Sequential(
        #         *garnet_layers
        # )

        self._pred_layers = nn.Sequential(
                nn.Linear(2*prev_dim, prev_dim),
                nn.ReLU(),
                nn.Linear(prev_dim, num_classes)
        )

        self._att_layers = nn.Sequential(
                nn.Linear(prev_dim, prev_dim),
                nn.ReLU(),
                nn.Linear(prev_dim, 1),
        )
        self.prev_dim = prev_dim

    def forward(self, X):
        for layer in self._layers:
            pred_x, agg = layer(X)
            X = pred_x
        mean_pooled = torch.mean(pred_x, axis=-2)
        # max_pooled = torch.max(pred_x, axis=-2)[0]
        max_pooled = torch.max(pred_x, axis=-2)[1]
        # atten_pooled = torch.sum(torch.softmax(self._axtt_layers(pred_x), 1).repeat(1, 1, self.prev_dim)*pred_x, axis=-2)
        H = torch.cat((mean_pooled, max_pooled), axis=-1)
        # H = mean_pooled
        # H = max_pooled
        # H = atten_pooled
        return self._pred_layers(H)


class Bipartite_Layers(nn.Module):
    def __init__(self,
        input_dim,
        feature_dim,
        n_aggregators,
        hidden_activation,
        aggregator_activation,
        num_heads=4,
        ln=False
    ):
        super(Bipartite_Layers, self).__init__()
        self.aggregator_activation = aggregator_activation
        self.enc = nn.Sequential(
            SAB(input_dim, feature_dim, num_heads, ln=ln),
            SAB(feature_dim, feature_dim, num_heads, ln=ln),
        )
        self.transform_in = nn.Linear(feature_dim, feature_dim)
        self.aggregator_score = nn.Linear(feature_dim, n_aggregators)
        self.transform_out = nn.Sequential(
                nn.Linear(2*feature_dim*n_aggregators + input_dim + feature_dim, feature_dim),
                getattr(nn, hidden_activation)()
        )
        self._feature_dim = feature_dim
        self._n_aggregators = n_aggregators

    def forward(self, X):
        """
        X: Tensor of shape [n_minibatches, n_tracks, n_track_features]
        """
        Xp = self.enc(X)
        Xp = self.transform_in(Xp)
        attention_score = self.aggregator_score(Xp)
        if self.aggregator_activation == 'potential':
            attention_score = torch.exp(-torch.abs(attention_score))
        elif self.aggregator_activation == 'ReLU':
            act = nn.ReLU()
            attention_score = act(attention_score)
        elif self.aggregator_activation == 'Tanh':
            act = nn.Tanh()
            attention_score = act(attention_score)
        elif self.aggregator_activation == 'softmax':
            attention_score = torch.softmax(attention_score, 1)
        else:
            attention_score = attention_score
        edges = torch.einsum('btf,bta->baft', Xp, attention_score)
        max_pooled = torch.max(edges, dim=-1)[0] # (n_minibatches, n_aggregators, n_features)
        mean_pooled = torch.mean(edges, dim=-1)
        aggregator = torch.cat((max_pooled, mean_pooled), axis=-1)
        aggregator = aggregator.reshape(aggregator.shape[0], aggregator.shape[1]*aggregator.shape[2])
        agg_repeated = aggregator.reshape(aggregator.shape[0], 1, -1).repeat(1, X.shape[1], 1)
        H = torch.cat((X, Xp, agg_repeated), axis=-1)
        return self.transform_out(H), aggregator
