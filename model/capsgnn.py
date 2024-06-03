import torch.nn
import time

from model.capsule_layer import firstCapsuleLayer, SecondaryCapsuleLayer
from model.loss import adj_recons_loss
from model.resconstruct_layer import ReconstructionLayer
from utils import cuda_device, sizee, stop


class CapsGNN(torch.nn.Module):
    def __init__(self, args, max_node_num, number_of_features, number_of_targets):
        super(CapsGNN, self).__init__()
        self.args = args
        self.max_node_num = max_node_num
        self.number_of_features = number_of_features
        self.number_of_targets = number_of_targets
        self._setup_layers()

    def _setup_layers(self):
        self._setup_firstCapsuleLayer()
        self._setup_hiddenCapsuleLayer()
        self._setup_classCapsuleLayer()
        self._setup_reconstructNet()

    def _setup_firstCapsuleLayer(self):
        self.first_capsule = firstCapsuleLayer(number_of_features=self.number_of_features,
                                               features_dimensions=self.args.features_dimensions,
                                               capsule_dimensions=self.args.capsule_dimensions,
                                               num_gcn_layers=self.args.num_gcn_layers,
                                               num_gcn_channels=self.args.num_gcn_channels,
                                               dropout=self.args.dropout)

    def _setup_hiddenCapsuleLayer(self):
        self.hidden_capsule = SecondaryCapsuleLayer(k=self.args.k,
                                                    batch_size=self.args.batch_size,
                                                    num_iterations=self.args.num_iterations,
                                                    low_num_capsules=self.max_node_num * self.args.num_gcn_layers,
                                                    num_capsules=self.args.capsule_num,
                                                    in_cap_dim=self.args.capsule_dimensions,
                                                    out_cap_dim=self.args.capsule_dimensions,
                                                    num_gcn_layers=self.args.num_gcn_layers,
                                                    dropout=self.args.dropout)

    def _setup_classCapsuleLayer(self):
        self.class_capsule = SecondaryCapsuleLayer(k=self.args.k,
                                                   batch_size=self.args.batch_size,
                                                   num_iterations=self.args.num_iterations,
                                                   low_num_capsules=self.args.capsule_num,
                                                   num_capsules=self.number_of_targets,
                                                   in_cap_dim=self.args.capsule_dimensions,
                                                   out_cap_dim=self.args.capsule_dimensions,
                                                   num_gcn_layers=self.args.num_gcn_layers,
                                                   dropout=self.args.dropout)

    def _setup_reconstructNet(self):
        self.recons_net = ReconstructionLayer(n_dim=self.args.capsule_dimensions ** 2,
                                              n_classes=self.number_of_targets,
                                              hidden=self.args.capsule_dimensions ** 2,
                                              k=self.args.k2)
    # ## CapsGNN
    # def _setup_reconstructNet(self):
    #     self.recons_net = ReconstructionLayer(n_dim=self.args.capsule_dimensions ** 2,
    #                                           n_classes=self.number_of_targets,
    #                                           hidden=self.number_of_features)

    def forward(self, x, adj, y, mask):
        epsilon = 1e-7
        batch_size = x.size(0)
        first_out = self.first_capsule(x, adj, mask)
        second_out, second_adj = self.hidden_capsule(first_out, adj, mask)
        second_out = second_out.squeeze(4).squeeze(1)
        class_out, class_adj = self.class_capsule(second_out, second_adj)
        class_out = class_out.squeeze(4).squeeze(1)
        residual = first_out.view(batch_size, x.size(1), -1, self.args.capsule_dimensions ** 2)[:, :, -1, :]
        
        
        # LightCapsGNN
        recons_out = self.recons_net(residual, second_out, class_out, y, mask)
        recon_loss = adj_recons_loss(recons_out, adj, mask)
        # recons_out = self.recons_net(residual, class_out, y, mask)
        # recon_loss = self.recons_net(x, class_out, y, mask)

        out = (torch.sqrt((class_out ** 2).sum(2) + epsilon)).view(batch_size, self.number_of_targets)

        # max_out, _ = out.max(dim=1)
        # l1_loss = -torch.log(max_out)
        # l1_loss = l1_loss.mean()
        # out = (torch.sqrt((class_out ** 2).sum(2) + epsilon)).view(batch_size, self.number_of_targets)
        # return out, recon_loss,l1_loss
        return out, recon_loss
