import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

class GAE(nn.Module):
    def __init__(self, in_dim, num_hidden, num_classes, dropout):
        super(GAE, self).__init__()
        self.gc1=GraphConv(in_dim, num_hidden, activation=activation))
        # self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(num_hidden, num_hidden, activation=activation)
        # self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        # hidden1 = self.gc1(x, adj)
        hidden1 = self.gc1(self.g, self.dropout(x))
        hidden2 = self.gc2(self.g, self.dropout(hidden1))
        return hidden2

    def forward(self, x, adj):
        hidden = self.encode(x)
        return self.dc(hidden)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj