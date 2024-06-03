from utils import cuda_device, sizee
import torch
from torch import nn
import torch.nn.functional as F


def margin_loss(scores, target, loss_lambda=0.5):
    target = F.one_hot(target, scores.size(1))
    v_mag = scores
    zero = torch.zeros(1)
    zero = cuda_device(zero)
    m_plus = 0.9
    m_minus = 0.1

    max_l = torch.max(m_plus - v_mag, zero) ** 2
    max_r = torch.max(v_mag - m_minus, zero) ** 2
    T_c = target

    L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
    L_c = L_c.sum(dim=1)
    L_c = L_c.mean()
    # print("margin loss",L_c)
    return L_c


def adj_recons_loss(pred_adj, adj, mask=None):
    eps = 1e-7
    # Each entry in pred_adj cannot larger than 1
    pred_adj = torch.min(pred_adj, cuda_device(torch.ones(1, dtype=pred_adj.dtype)))
    # The diagonal entries in pred_adj should be 0
    pred_adj = pred_adj.masked_fill_(cuda_device(torch.eye(adj.size(1), adj.size(1)).bool()), 0)
    # Cross entropy loss
    link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
    if mask is not None:
        num_entries = torch.sum(torch.sum(mask, dim=1) ** 2)
        adj_mask = mask.unsqueeze(2).float() @ torch.transpose(mask.unsqueeze(2).float(), 1, 2)
        link_loss[(1 - adj_mask).bool()] = 0.0
    else:
        num_entries = pred_adj.size(0) * pred_adj.size(1) * pred_adj.size(2)

    link_loss = torch.sum(link_loss) / float(num_entries)
    # print("link loss", link_loss)
    return link_loss
