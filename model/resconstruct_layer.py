import torch
from torch import topk

from utils import cuda_device, sizee, stop
import torch.nn as nn
import torch.nn.functional as F
import time


def sparse2dense(x, new_size, mask):
    out = cuda_device(torch.zeros(new_size))
    out[mask] = x
    return out


def squash(input_tensor, dim=-1, epsilon=1e-11):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = input_tensor / safe_norm
    return unit_vector, scale



# our ReconstructionLayer
class ReconstructionLayer(torch.nn.Module):
    def __init__(self, n_dim, n_classes, hidden,k):
        super(ReconstructionLayer, self).__init__()
        self.n_dim = n_dim
        self.n_classes = n_classes
        # self.fc1 = nn.Linear(n_dim, hidden)
        self.fc1 = nn.Linear(n_dim, hidden)
        self.k=k
        # self.fc2 = nn.Linear(n_dim * 2, hidden)

    def forward(self, first_capsule, second_capsule, class_capsule, y, first_capsule_mask):
        mask = torch.zeros((class_capsule.size(0), self.n_classes))
        mask = cuda_device(mask)
        mask.scatter_(1, y.view(-1, 1), 1.)
        mask1 = mask.bool()
        mask = mask.unsqueeze(2)
        class_capsule = class_capsule[mask1]
        class_capsule = class_capsule.view(-1, 1, self.n_dim)

        class_capsule, scale1 = squash(class_capsule)

        second_capsule, scale = squash(second_capsule)

        _, indice = topk(scale, k=self.k, dim=1)
        # class_capsule=class_capsule[indice.squeeze(-1)]
        second_capsule = torch.take_along_dim(second_capsule, indice, dim=1)
        scale = torch.take_along_dim(scale, indice, dim=1)

        second_capsule = torch.cat((second_capsule, class_capsule), dim=1)
        scale = torch.cat((scale, scale1), dim=1)
        # combine the first capsule and the class capsule (class-conditional)
        # W*mask(class_caps)+b
        second_capsule = F.relu(self.fc1(second_capsule))
        first_capsule = F.relu(first_capsule)
        x = first_capsule.unsqueeze(1) + second_capsule.unsqueeze(2)

        # x = torch.cat((first_capsule, class_capsule.repeat(1, first_capsule.size(1), 1)), dim=-1)
        # x = F.relu(self.fc2(x))
        # x = x * first_capsule_mask.unsqueeze(-1)

        # torch.transpose(x, 2, 1)
        adj = torch.matmul(x, torch.transpose(x, -2, -1))
        adj = adj * scale.unsqueeze(-1)
        adj = adj.sum(dim=1)
        adj_1 = adj[first_capsule_mask]
        adj = sparse2dense(adj_1, (adj.size(0), adj.size(1), adj.size(2)), first_capsule_mask)
        return adj



# CapsGNN ReconstructionLayer
# class ReconstructionLayer(torch.nn.Module):
#     def __init__(self, n_dim, n_classes, hidden):
#         super(ReconstructionLayer, self).__init__()
#         self.n_dim = n_dim
#         self.n_classes = n_classes
#         self.hidden = hidden
#         # self.fc1 = nn.Linear(n_dim, hidden)
#         self.fc1 = nn.Linear(self.n_dim, self.n_dim // 2)
#         self.fc2 = nn.Linear(self.n_dim // 2, hidden)

#     def forward(self, first_capsule, class_capsule, y, first_capsule_mask):
#         mask = torch.zeros((class_capsule.size(0), self.n_classes))
#         mask = cuda_device(mask)
#         mask.scatter_(1, y.view(-1, 1), 1.)
#         mask1 = mask.bool()
#         mask = mask.unsqueeze(2)
#         class_capsule = class_capsule[mask1]
#         class_capsule = class_capsule.view(-1, 1, self.n_dim)

#         decoded = self.fc1(class_capsule)
#         decoded = self.fc2(decoded)
#         decoded = torch.sigmoid(decoded)

#         pos_mask = torch.where(first_capsule != 0, 1., 0.)
#         pos_mask = pos_mask * first_capsule_mask.unsqueeze(-1)

#         tmp1 = pos_mask.sum(dim=1)
#         mask_1 = torch.where(tmp1 > 0, 1., 0.)
#         mask_0 = 1 - mask_1
#         nodes = torch.sum(first_capsule_mask, dim=-1)
#         pos = torch.sum(mask_1 * (decoded.squeeze(1) - tmp1) ** 2, dim=-1) / torch.sum(mask_1, dim=-1)
#         neg = torch.sum(mask_0 * (decoded.squeeze(1) - tmp1) ** 2, dim=-1) / torch.sum(mask_0, dim=-1)

#         Loss = (pos + neg)/nodes

#         return Loss.mean()
