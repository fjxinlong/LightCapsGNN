import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch.nn import Parameter

from model.attention import Attention
from model.denseGCNConv import DenseGCNConv
import torch_geometric.nn as pyg

from model.disentangle import linearDisentangle
from utils import cuda_device, sizee, stop


def squash(input_tensor, dim=-1, epsilon=1e-11):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale * unit_vector, scale


def sparse2dense(x, new_size, mask):
    out = cuda_device(torch.zeros(new_size))
    out[mask] = x
    return out


def routing(u_hat, num_iteration, mask=None):
    u_hat_size = u_hat.size()
    b_ij = torch.zeros(u_hat_size[0], u_hat_size[1], u_hat_size[2], 1, 1)
    # _, b_ij = squash(u_hat, dim=-2)
    b_ij = cuda_device(b_ij)  # [bs,n,upper_caps,1]

    for i in range(num_iteration - 1):
        c_ij = F.softmax(b_ij, dim=2)  # [bs,n*(neighbor+1),upper_caps,1,1]
        s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # [bs,1,upper_caps,d,1,1]
        v, a_j = squash(s_j, dim=-2)
        u_produce_v = torch.matmul(u_hat.transpose(-1, -2), v)  # [bs,n*(neighbor+1),upper_caps,1,1]
        b_ij = b_ij + u_produce_v  # [bs,n*(neighbor+1),upper_caps,1,1]
    if mask is not None:
        b_ij = b_ij * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, b_ij.size(2),
                                                                            b_ij.size(3),
                                                                            b_ij.size(4))
    c_ij = F.softmax(b_ij, dim=2)  # [bs,n*(neighbor+1),upper_caps,1,1]
    # c_ij = F.sigmoid(b_ij)  # [bs,n*(neighbor+1),upper_caps,1,1]
    return c_ij



# # capsGNN
# class firstCapsuleLayer(torch.nn.Module):
#     def __init__(self, number_of_features, features_dimensions, capsule_dimensions, num_gcn_layers, num_gcn_channels,
#                  dropout):
#         super(firstCapsuleLayer, self).__init__()

#         self.number_of_features = number_of_features
#         self.features_dimensions = features_dimensions
#         self.capsule_dimensions = capsule_dimensions
#         self.dropout = nn.Dropout(p=dropout)
#         self.num_gcn_layers = num_gcn_layers
#         self.num_gcn_channels = num_gcn_channels
#         self.gcn_layers_dims = self.capsule_dimensions ** 2

#         # self.bn = nn.BatchNorm1d(self.number_of_features)
#         self.encapsule = nn.ModuleList()
#         for i in range(self.capsule_dimensions):
#             self.encapsule.append(
#                 linearDisentangle(self.capsule_dimensions, self.capsule_dimensions))
#         # GCN
#         self.gcn_layers = nn.ModuleList()
#         self.gcn_layers.append(DenseGCNConv(self.number_of_features, self.capsule_dimensions))
#         for _ in range(self.num_gcn_layers-1):
#             self.gcn_layers.append(DenseGCNConv(self.capsule_dimensions, self.capsule_dimensions))

#         self.attention = Attention(self.gcn_layers_dims * self.num_gcn_layers, self.num_gcn_layers)

#     def forward(self, x, adj, mask):

#         x_size = x.size()
#         features=x
#         hidden_representations = []

#         for layer in self.gcn_layers:
#             features = layer(features, adj, mask)
#             features = torch.tanh(features)
#             features = self.dropout(features)
#             hidden_representations.append(features.reshape(x_size[0], x_size[1], 1, -1))
#         hidden_representations = torch.cat(hidden_representations, dim=2)

#         x = hidden_representations[mask]  # (N1+N2+...+Nm)*d
#         # print("11:",x.size())

#         out = []
#         # x = self.bn(x)

#         for i, encaps in enumerate(self.encapsule):
#             temp = F.relu(encaps(x))
#             # temp = self.dropout(temp)
#             out.append(temp)

#         out = torch.cat(out, dim=-1)
#         out = sparse2dense(out, (x_size[0], x_size[1], out.size(-2),out.size(-1)), mask)
#         out, _ = squash(out)
#         features = out

#         hidden_representations = out
#         attn = self.attention(hidden_representations.reshape(x_size[0], x_size[1], -1))
#         attn = F.softmax(attn.masked_fill(mask.unsqueeze(-1).eq(0), 0), dim=-1)

#         # attn1 = attn.masked_fill(mask.unsqueeze(-1).eq(0), 0)
#         # attn1 = attn1.unsqueeze(-1)
#         # number_of_nodes = torch.sum(mask, dim=1, keepdim=True).float().unsqueeze(-1)
#         # # hidden_representations = hidden_representations * attn * number_of_nodes
#         hidden_representations = hidden_representations * attn.unsqueeze(-1)
#         return hidden_representations.reshape(x_size[0], -1, self.gcn_layers_dims)

#our method
class firstCapsuleLayer(torch.nn.Module):
    def __init__(self, number_of_features, features_dimensions, capsule_dimensions, num_gcn_layers, num_gcn_channels,
                 dropout):
        super(firstCapsuleLayer, self).__init__()

        self.number_of_features = number_of_features
        self.features_dimensions = features_dimensions
        self.capsule_dimensions = capsule_dimensions
        self.dropout = nn.Dropout(p=dropout)
        self.num_gcn_layers = num_gcn_layers
        self.num_gcn_channels = num_gcn_channels
        self.gcn_layers_dims = self.capsule_dimensions ** 2

        # self.bn = nn.BatchNorm1d(self.number_of_features)
        print(self.number_of_features)
        # 0/0
        self.encapsule = nn.ModuleList()
        for i in range(self.capsule_dimensions):
            self.encapsule.append(
                linearDisentangle(self.number_of_features, self.capsule_dimensions))
        # GCN
        self.gcn_layers = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            self.gcn_layers.append(DenseGCNConv(self.gcn_layers_dims, self.gcn_layers_dims))

        self.attention = Attention(self.gcn_layers_dims * self.num_gcn_layers, self.num_gcn_layers)

    def forward(self, x, adj, mask):

        x_size = x.size()
        x = x[mask]  # (N1+N2+...+Nm)*d
        out = []
        # x = self.bn(x)

        for i, encaps in enumerate(self.encapsule):
            temp = F.relu(encaps(x))
            # temp = self.dropout(temp)
            out.append(temp)

        out = torch.cat(out, dim=-1)
        out = sparse2dense(out, (x_size[0], x_size[1], out.size(-1)), mask)
        out, _ = squash(out)
        features = out

        hidden_representations = []
        for layer in self.gcn_layers:
            features = layer(features, adj, mask)
            features = torch.tanh(features)
            features = self.dropout(features)
            hidden_representations.append(features.reshape(x_size[0], x_size[1], 1, -1))
        hidden_representations = torch.cat(hidden_representations, dim=2)

        # hidden_representations = hidden_representations
        attn = self.attention(hidden_representations.reshape(x_size[0], x_size[1], -1))
        attn = F.softmax(attn.masked_fill(mask.unsqueeze(-1).eq(0), 0), dim=-1)
        # attn1 = attn.masked_fill(mask.unsqueeze(-1).eq(0), 0)
        # attn1 = attn1.unsqueeze(-1)
        # number_of_nodes = torch.sum(mask, dim=1, keepdim=True).float().unsqueeze(-1)
        # # hidden_representations = hidden_representations * attn * number_of_nodes
        hidden_representations = hidden_representations * attn.unsqueeze(-1)
        return hidden_representations.reshape(x_size[0], -1, self.gcn_layers_dims)

# CapsGNN
# class SecondaryCapsuleLayer(torch.nn.Module):
#     def __init__(self, k,batch_size,num_iterations, num_capsules, low_num_capsules, in_cap_dim, out_cap_dim, num_gcn_layers,
#                  dropout):
#         super(SecondaryCapsuleLayer, self).__init__()
#         self.num_iterations = num_iterations
#         self.num_higher_cap = num_capsules  
#         self.num_lower_cap = low_num_capsules  
#         self.num_gcn_layers = num_gcn_layers
#         self.in_cap_dim = in_cap_dim
#         self.out_cap_dim = out_cap_dim
#         self.dropout = nn.Dropout(p=dropout)
#         self.bn = nn.BatchNorm1d(self.in_cap_dim)
#         self.W = torch.nn.Parameter(
#             torch.randn(self.num_lower_cap, self.num_higher_cap, self.in_cap_dim*in_cap_dim, self.out_cap_dim*out_cap_dim))
#         # self.W = torch.nn.Parameter(torch.randn(self.num_higher_cap, self.in_cap_dim, self.out_cap_dim))

#     def forward(self, x, adj, mask=None):
#         x_size, batch_size, max_node_num = x.size(), x.size(0), x.size(1)
#         if mask is not None:
#             mask = mask.repeat(1, self.num_gcn_layers)
#         # if mask is not None:
#         #     x = x[mask]  # (N1+N2+...+Nm)*d
#         #     # x = self.bn(x)
#         #     x = sparse2dense(x, x_size, mask)  # (B,N,d)
#         #     x = x.reshape(batch_size, max_node_num, 1, 1, -1)
#         # else:
#         #     x = x.reshape(-1, self.in_cap_dim)
#         #     # x = self.bn(x)
#         #     x = x.reshape(batch_size, max_node_num, 1, 1, -1)
#         # x = x.reshape(batch_size, max_node_num, 1, self.in_cap_dim, self.in_cap_dim)
#         x = x.reshape(batch_size, max_node_num, 1, 1,-1)
#         x = x.repeat(1, 1, self.num_higher_cap, 1,1)
#         # W = self.W.repeat(batch_size, max_node_num, 1, 1)
#         W = self.W
#         # W = W.repeat(x.size(1), 1, 1, 1)
#         # u_hat = torch.matmul(W, x).unsqueeze(4)
#         # print(x.size())
#         # print(W.size())
#         # 0/0
       
#         # print(self.num_higher_cap)
#         # print(self.num_lower_cap)
#         u_hat = torch.matmul(x, W)
#         u_hat = u_hat.reshape(batch_size, max_node_num, self.num_higher_cap, -1, 1)
#         if mask is not None:
#             u_hat = u_hat * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, u_hat.size(2),
#                                                                                   u_hat.size(3),
#                                                                                   u_hat.size(4))
#         # u_hat = torch.matmul(W, x)
#         temp_u_hat = u_hat.detach()
#         c_ij = routing(temp_u_hat, num_iteration=self.num_iterations, mask=mask)

#         s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # [bs,1,upper_caps,d,1]
#         v, a_j = squash(s_j, dim=-2)
#         # v,a_j = squash(s_j, dim=-2).squeeze(4).squeeze(1)
#         # v = v[:, :, :-3, :, :]

#         return v, a_j

# # TR
# class SecondaryCapsuleLayer(torch.nn.Module):
#     def __init__(self, k, batch_size, num_iterations, num_capsules, low_num_capsules, in_cap_dim, out_cap_dim,
#                  num_gcn_layers,
#                  dropout):
#         super(SecondaryCapsuleLayer, self).__init__()
#         self.num_iterations = num_iterations
#         self.num_higher_cap = num_capsules  
#         self.num_lower_cap = low_num_capsules  
#         self.num_gcn_layers = num_gcn_layers
#         self.in_cap_dim = in_cap_dim
#         self.out_cap_dim = out_cap_dim
#         self.dropout = nn.Dropout(p=dropout)
#         self.bn = nn.BatchNorm1d(self.in_cap_dim)
#         self.k = k
#         self.batch_size = batch_size
#         # self.W = torch.nn.Parameter(
#         #     torch.randn(self.num_lower_cap, self.num_higher_cap, self.in_cap_dim, self.out_cap_dim))
#         self.W = torch.nn.Parameter(torch.randn(self.k, self.in_cap_dim, self.out_cap_dim))
#         self.alpha = torch.nn.Parameter(torch.randn(self.num_lower_cap, self.num_higher_cap, self.k))
#         self.beta = torch.nn.Parameter(
#             torch.randn(self.batch_size, self.num_lower_cap, self.num_higher_cap))

#     def forward(self, x, adj, mask=None):
#         x_size, batch_size, max_node_num = x.size(), x.size(0), x.size(1)
#         if mask is not None:
#             mask = mask.repeat(1, self.num_gcn_layers)
#             x = x * mask.unsqueeze(-1)
#         # x = x.reshape(batch_size, max_node_num, 1, self.in_cap_dim, self.in_cap_dim)

#         alpha = self.alpha
#         beta = self.beta
#         if batch_size < self.batch_size:
#             # alpha = alpha[:batch_size, :, :, :]
#             beta = beta[:batch_size, :, :]

#         tmp1 = beta.unsqueeze(-1) * alpha
#         tmp2 = tmp1.unsqueeze(-1) * x.unsqueeze(-2).unsqueeze(-2)
#         tmp3 = tmp2.sum(dim=1)
#         tmp4 = torch.matmul(tmp3.view(batch_size, self.num_higher_cap, self.k, self.in_cap_dim, -1), self.W)
#         tmp5 = tmp4.view(batch_size, self.num_higher_cap, self.k, -1)
#         tmp6 = tmp5.sum(dim=2)
#         # x = x.repeat(1, 1, self.num_higher_cap, 1, 1)
#         # # W = self.W.repeat(batch_size, max_node_num, 1, 1)
#         # W = self.W
#         # # W = W.repeat(x.size(1), 1, 1, 1)
#         # # u_hat = torch.matmul(W, x).unsqueeze(4)
#         # u_hat = torch.matmul(x, W)
#         # u_hat = u_hat.reshape(batch_size, max_node_num, self.num_higher_cap, -1, 1)
#         # if mask is not None:
#         #     u_hat = u_hat * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, u_hat.size(2),
#         #                                                                           u_hat.size(3),
#         #                                                                           u_hat.size(4))
#         # # u_hat = torch.matmul(W, x)
#         # temp_u_hat = u_hat.detach()
#         # c_ij = routing(temp_u_hat, num_iteration=self.num_iterations, mask=mask)
#         #
#         # s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # [bs,1,upper_caps,d,1]
#         s_j = tmp6.unsqueeze(1).unsqueeze(4)
#         v, a_j = squash(s_j, dim=-2)
#         # v,a_j = squash(s_j, dim=-2).squeeze(4).squeeze(1)
#         # v = v[:, :, :-3, :, :]
#         return v, a_j

# DR
class SecondaryCapsuleLayer(torch.nn.Module):
    def __init__(self, k, batch_size, num_iterations, num_capsules, low_num_capsules, in_cap_dim, out_cap_dim,
                 num_gcn_layers,
                 dropout):
        super(SecondaryCapsuleLayer, self).__init__()
        self.num_iterations = num_iterations
        self.num_higher_cap = num_capsules  
        self.num_lower_cap = low_num_capsules  
        self.num_gcn_layers = num_gcn_layers
        self.in_cap_dim = in_cap_dim
        self.out_cap_dim = out_cap_dim
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(self.in_cap_dim)
        self.k = k
        self.batch_size = batch_size
        # self.W = torch.nn.Parameter(
        #     torch.randn(self.num_lower_cap, self.num_higher_cap, self.in_cap_dim, self.out_cap_dim))
        self.W = torch.nn.Parameter(torch.randn(self.k, self.in_cap_dim, self.out_cap_dim))
        self.alpha = torch.nn.Parameter(torch.randn(self.num_lower_cap, self.num_higher_cap, self.k))
        # self.gcn = DenseGCNConv(self.in_cap_dim ** 2, self.out_cap_dim ** 2)

    def forward(self, x, adj, mask=None):
  
        x_size, batch_size, max_node_num = x.size(), x.size(0), x.size(1)
        if mask is not None:
            mask = mask.repeat(1, self.num_gcn_layers)
        # x = x.reshape(batch_size, max_node_num, 1, self.in_cap_dim, self.in_cap_dim)
        alpha = self.alpha
        tmp1 = self.W.unsqueeze(0).unsqueeze(0).unsqueeze(0) * alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        tmp2  = tmp1.sum(dim=3)
        x = x.view(batch_size, self.num_lower_cap, 1, self.in_cap_dim, -1)
        tmp4 = torch.matmul(x, tmp2)
        # x=x.view(batch_size, self.num_lower_cap, 1, self.in_cap_dim, -1)
        # tmp1=torch.matmul(x,self.W.unsqueeze(0).unsqueeze(0)).unsqueeze(2)*alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # tmp4=tmp1.sum(dim=3)
        u_hat = tmp4
        u_hat = u_hat.view(batch_size, max_node_num, self.num_higher_cap, -1, 1)
        if mask is not None:
            u_hat = u_hat * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, u_hat.size(2),
                                                                                  u_hat.size(3),
                                                                                  u_hat.size(4))
        # u_hat = torch.matmul(W, x)
        temp_u_hat = u_hat.detach()
        c_ij = routing(temp_u_hat, num_iteration=self.num_iterations, mask=mask)

        s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # [bs,1,upper_caps,d,1]
        v, a_j = squash(s_j, dim=-2)
        # v,a_j = squash(s_j, dim=-2).squeeze(4).squeeze(1)
        # v = v[:, :, :-3, :, :]
        # if mask is not None:
        #     c_ij = c_ij.view(batch_size, adj.size(1), -1, self.num_higher_cap)  # [bs,n ,(neighbor+1),upper_caps]
        #     c_ij = torch.mean(c_ij, dim=2)  # [bs,n ,upper_caps]
        #     adj = torch.transpose(c_ij, 2, 1) @ adj @ c_ij  # [bs,upper_caps,upper_caps]
        #     adj = F.softmax(adj, dim=2)
     

        return v, a_j
