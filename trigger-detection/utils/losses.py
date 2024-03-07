import torch
import torch.nn.functional as F
from torch import nn
from icecream import ic

__all__ = ['SupCon', 'sup_con']


class SupCon(nn.Module):
    """
    Calculates the SupCon loss for supervised contrastive learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_key are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        positive_key: (N, M, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_key (N, M, D): Tensor with negative samples (e.g. embeddings of other inputs)
    Returns:
         Value of the SupCon Loss.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, positive_key, negative_key):
        return sup_con(query, positive_key, negative_key, temperature=self.temperature)

def sup_con(query, positive_keys, negative_keys, temperature=0.1):
    # TODO: can possibly be made more efficient by removing the for-loop
    # but right now correctness is our desire
    positive_logits = torch.einsum('bf,bjf->bj', query, positive_keys)
    negative_logits = torch.einsum('bf,bjf->bj', query, negative_keys)

    labels = torch.zeros(len(query), dtype=torch.long, device=query.device)
    total_loss = 0
    for i in range(positive_keys.shape[1]):
        logits = torch.cat([positive_logits[:, i].unsqueeze(-1), negative_logits], dim=-1)
        total_loss += F.cross_entropy(logits / temperature, labels, reduction='mean')
    return total_loss / positive_keys.shape[1]

# query is expected to be (B, F)
# negative_key is expected to be (B, N_N, F)
# positive keys is expected to be (B, N_P, F)
def sup_con_old_2(query, positive_key, negative_key, temperature=0.1):
    positive_logit = torch.einsum('bf,bjf->bj', query, positive_key).unsqueeze(1).repeat(1, negative_key.shape[1], 1)
    # pl_bj: similarity between query b and positive example j
    negative_logit = torch.einsum('bf,bjf->bj', query, negative_key).unsqueeze(2).repeat(1, 1, positive_key.shape[1])

    # l is (B, N_P)
    #print(negative_logit.shape)
    #print(positive_logit.shape)
    #print(torch.exp(negative_logit - positive_logit).shape)
    ic((negative_logit - positive_logit)/temperature)
    ic(torch.exp(torch.exp((negative_logit - positive_logit)/temperature)))
    ic(torch.sum(torch.exp((negative_logit - positive_logit)/temperature), dim=1))
    ic(1/(torch.sum(torch.exp((negative_logit - positive_logit)/temperature), dim=1) + 1e-16))
    l = 1/(torch.sum(torch.exp((negative_logit - positive_logit)/temperature), dim=1) + 1e-16)
    ic(torch.log(l + 1e-16))
    ic(l)
    pmb_loss = torch.sum(torch.log(l + 1e-16), dim=-1)
    ic(pmb_loss)
    loss = -1/positive_key.shape[1] * torch.sum(pmb_loss)

    return loss
def sup_con_old(query, positive_key, negative_key, temperature=0.1):
    ic(query)
    ic(positive_key)
    ic(negative_key)
    positive_logit = torch.einsum('bf,bjf->bj', query, positive_key)
    # pl_bj: similarity between query b and positive example j
    negative_logit = torch.einsum('bf,bjf->bj', query, negative_key)
    # nl_bj: similarity between query b and negative example j
    mp = torch.max(positive_logit, dim=-1)[0]
    # mp_b: maximum similarity between query b and any positive example
    print('Before')
    ic(positive_logit)
    ic(negative_logit)
    print('After')
    # positive logit shape: (B, N_P)
    # negative logit shape (B, N_N)
    positive_logit = torch.exp((positive_logit - mp.unsqueeze(-1))/temperature) + 1e-16
    negative_logit = torch.exp((negative_logit - mp.unsqueeze(-1))/temperature)
    ic(positive_logit)
    ic(negative_logit)

    negative_logit = torch.sum(negative_logit, dim=-1)
    # nl_b: total sum of similarity scores between all negative examples
    # nl_bi

    # Reduce pl_bj -> pl_b by summing across all positive examples. 
    ic(positive_logit)
    ic(negative_logit + 1e-16)
    ic(positive_logit/(negative_logit.unsqueeze(-1) + 1e-16))
    pmb_loss = torch.sum(torch.log(positive_logit/(negative_logit.unsqueeze(-1) + 1e-16)), dim=-1)
    ic(pmb_loss)
    loss = 1/positive_key.shape[1] * torch.sum(pmb_loss)

    return loss
