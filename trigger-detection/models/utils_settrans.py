import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .layers import Masked_LayerNorm

from icecream import ic
import sys

class Masked_MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        if ln:
            self.ln0 = Masked_LayerNorm(dim_V)
            self.ln1 = Masked_LayerNorm(dim_V)


    def forward(self, Q, K, mask):
        def masked_softmax(X):
            X_orig = X
            # head*batch, track, feature <- 2
            # new: head, batch, track, feature <- 3
            # head, batch, track, feature
            X = X.reshape(self.num_heads, Q.shape[0], X.shape[1], X.shape[2])
            max_elem = torch.max(X, dim=3)[0]
            mask_1 = mask.reshape(1, mask.shape[0], mask.shape[1], 1)
            mask_2 = mask.reshape(1, mask.shape[0], 1, mask.shape[1])
            assert not torch.any(torch.isnan(max_elem))

            X_temp = (X - max_elem.unsqueeze(3)) * mask_1 * mask_2
            num = torch.exp(X_temp)
            num = num * mask_1 * mask_2
            dem = torch.sum(num, dim=3).unsqueeze(3)
            res = num/(dem + 1e-16)
            res = res.reshape(X_orig.shape[0], X_orig.shape[1], X_orig.shape[2])
            return res

        Q = self.fc_q(Q)*mask.unsqueeze(-1)
        K, V = self.fc_k(K)*mask.unsqueeze(-1), self.fc_v(K)*mask.unsqueeze(-1)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = masked_softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V))
        # Hopefully, for all padded values, A is 0
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O, mask)
        O = O + F.relu(self.fc_o(O)*mask.unsqueeze(-1))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O, mask)
        return O


class Masked_MAB_PT(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        if ln:
            self.ln0 = Masked_LayerNorm(dim_V)
            self.ln1 = Masked_LayerNorm(dim_V)


    def forward(self, Q, K, U, mask):
        def masked_softmax(X):
            X_orig = X
            # head*batch, track, feature <- 2
            # new: head, batch, track, feature <- 3
            # head, batch, track, feature
            X = X.reshape(self.num_heads, Q.shape[0], X.shape[1], X.shape[2])
            max_elem = torch.max(X, dim=3)[0]
            mask_1 = mask.reshape(1, mask.shape[0], mask.shape[1], 1)
            mask_2 = mask.reshape(1, mask.shape[0], 1, mask.shape[1])
            assert not torch.any(torch.isnan(max_elem))

            X_temp = (X - max_elem.unsqueeze(3)) * mask_1 * mask_2
            num = torch.exp(X_temp)
            num = num * mask_1 * mask_2
            dem = torch.sum(num, dim=3).unsqueeze(3)
            res = num/(dem + 1e-16)
            res = res.reshape(X_orig.shape[0], X_orig.shape[1], X_orig.shape[2])
            return res

        Q = self.fc_q(Q)*mask.unsqueeze(-1)
        K, V = self.fc_k(K)*mask.unsqueeze(-1), self.fc_v(K)*mask.unsqueeze(-1)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        # First:
        # (0, 1, 2, 3) -> (0, 3, 2, 1)
        # Second:
        # (0, 1, 2, 3) -> (0, 3, 1, 2)
        U = U.transpose(1, 3).transpose(-1, -2)
        U = U.reshape(U.shape[0]*U.shape[1], U.shape[2], U.shape[3])
        A = masked_softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V) + U)

        # Hopefully, for all padded values, A is 0
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O, mask)
        O = O + F.relu(self.fc_o(O)*mask.unsqueeze(-1))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O, mask)
        return O


class Masked_SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(Masked_SAB, self).__init__()
        self.mab = Masked_MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask):
        return self.mab(X, X, mask)

class Masked_SAB_PT(nn.Module):
    def __init__(self, dim_in, dim_out, ln=False):
        super().__init__()
        self.mab = Masked_MAB_PT(dim_in, dim_in, dim_out, num_heads=4, ln=ln)

    def forward(self, X, U, mask):
        return self.mab(X, X, U, mask)



class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
