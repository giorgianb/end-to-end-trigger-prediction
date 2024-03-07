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

# local
from .utils_settrans import Masked_SAB

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
            temperature=1
            ):
        super().__init__()
        from .Bipartite_Attention_Masked import Bipartite_Attention as BA
        self.temperature = temperature
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


        act = getattr(nn, hidden_activation)
        self.self_split = self_split
        if self.self_split:
            assert final_pooling, "We have not implement final_pooling=False when splitting"
            self.pooler = BA(
                    num_features=num_features,
                    num_classes=layers_spec[-1][0], 
                    layers_spec=layers_spec, 
                    hidden_activation=hidden_activation,
                    aggregator_activation=aggregator_activation,
                    ln=ln,
                    bn=bn,
                    recalculate_hits_mean=False,
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
                    num_features=layers_spec[-1][0],
                    num_classes=num_classes, 
                    layers_spec=layers_spec, 
                    hidden_activation=hidden_activation,
                    aggregator_activation=aggregator_activation,
                    ln=ln,
                    bn=bn,
                    recalculate_hits_mean=True,
                    self_split=False,
                    final_pooling=final_pooling
            )


        if ln:
            self.masker = nn.Sequential(
                    nn.Linear(layers_spec[-1][0], layers_spec[-1][0]),
                    nn.LayerNorm(layers_spec[-1][0]),
                    act(),
                    nn.Linear(layers_spec[-1][0], layers_spec[-1][0]),
                    nn.LayerNorm(layers_spec[-1][0]),
                    act(),
                    nn.Linear(layers_spec[-1][0], 2),
                )

            self.latent = nn.Sequential(
                    nn.Linear(num_features, layers_spec[-1][0]),
                    nn.LayerNorm(layers_spec[-1][0]),
                    act(),
                    nn.Linear(layers_spec[-1][0], layers_spec[-1][0]),
                    nn.LayerNorm(layers_spec[-1][0]),
                    act(),
                    nn.Linear(layers_spec[-1][0], layers_spec[-1][0]),
                )


        else:
            self.masker = nn.Sequential(
                    nn.Linear(layers_spec[-1][0], 64),
                    act(),
                    nn.Linear(64, 64),
                    act(),
                    nn.Sigmoid()
                )

            self.latent = nn.Sequential(
                    nn.Linear(num_features, layers_spec[-1][0]),
                    act(),
                    nn.Linear(layers_spec[-1][0], layers_spec[-1][0]),
                    act(),
                    nn.Linear(layers_spec[-1][0], layers_spec[-1][0]),
                )


    def forward(self, X, mask):
        if self.self_split:
            raise NotImplementedError("Need to implement masking correctly. Track Embeddings are are passed to the pooler instead of raw tracks.")
            hits = X[:, :, :15] * mask.unsqueeze(-1)
            hits = hits.reshape(X.shape[0], X.shape[1], 5, 3)

            # Generate new mask/hits mean in a particular quadrant
            hits_mask = hits[:, :, :, 0] >= 0
            Xp, maskp_1 = self.interpreter._mask_filter(X, mask, hits, hits_mask)
            Xp = self.interpreter._recalculate_hits_mean(Xp, maskp_1)

            # Generate track embeddings for that quadrant, and predict which tracks are kept
            track_embeddings = self.interpreter.generate_embedding(Xp, maskp_1)
            maskp_1_logits = self.masker(track_embeddings) * maskp_1.unsqueeze(-1)
            new_maskp_1 = F.gumbel_softmax(maskp_1_logits, hard=True, tau=self.temperature)
            new_maskp_1 = torch.argmax(new_maskp_1, dim=-1) * maskp_1

            # Pool track embeddings in that quadrant
            track_embeddings = track_embeddings * new_maskp_1.unsqueeze(-1)
            p1 = self.pooler(track_embeddings, new_maskp_1)

            # Generate new mask/hits mean in a particular quadrant
            hits_mask = hits[:, :, :, 0] < 0
            Xp, maskp_2 = self.interpreter._mask_filter(X, mask, hits, hits_mask)
            Xp = self.interpreter._recalculate_hits_mean(Xp, maskp_2)

            # Generate track embeddings for that quadrant, and predict which tracks are kept
            track_embeddings = self.interpreter.generate_embedding(Xp, maskp_2)
            maskp_2_logits = self.masker(track_embeddings) * maskp_2.unsqueeze(-1)
            new_maskp_2 = F.gumbel_softmax(maskp_2_logits, hard=True, tau=self.temperature)
            new_maskp_2 = torch.argmax(new_maskp_2, dim=-1) * maskp_2

            # Pool track embeddings in that quadrant
            track_embeddings = track_embeddings * new_maskp_2.unsqueeze(-1)
            p2 = self.pooler(track_embeddings, new_maskp_2)


            # Generate track embeddings for that quadrant, and predict which tracks are kept
            p = torch.cat([p1, p2], dim=-1)
            pred = self.classifier(p)

            # Combine the mask predictions
            mask_logits = torch.zeros(*mask.shape, 2, dtype=torch.float, device=mask.device)
            # We can combine them without worrying about masking since they are already masked
            mask_logits = maskp_1_logits + maskp_2_logits
        else:
            track_embeddings = self.interpreter.generate_embedding(X, mask)
            mask_logits = self.masker(track_embeddings) * mask.unsqueeze(-1)
            probs = F.gumbel_softmax(mask_logits, hard=False, tau=self.temperature)
            new_mask = probs[..., 1] * mask
            Xp = X * new_mask.unsqueeze(-1)
            pred = self.classifier(Xp, mask)

        return pred, mask_logits
