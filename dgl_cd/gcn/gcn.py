"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout=0.5, batch_norm=True, **kwargs):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, num_hidden, activation=activation))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(num_hidden, affine=False, track_running_stats=False)] * num_layers 

    def forward(self, features):
        h = features
        for idx, layer in enumerate(self.layers):
            if self.dropout !=0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if idx != len(self.layers) - 1:
                if self.batch_norm is not None:
                    h = self.batch_norm[idx](h)
        return h
    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]
    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]
