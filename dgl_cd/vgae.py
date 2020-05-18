import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class GVAE(nn.Module):
    def __init__(self, g, in_dim, num_hidden, num_classes, num_layers=1, dropout=0.5, *args, **kwargs):
        super(GVAE, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(p=dropout)
        self.gc1 = GraphConv(in_dim, num_hidden, activation=F.relu, bias=False)
        # hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.hidden_layers.append(
                GraphConv(num_hidden, num_hidden, activation=activation))
        self.gc2 = GraphConv(num_hidden, num_classes, bias=False)
        self.gc3 = GraphConv(num_hidden, num_classes, bias=False)
        self.batch_norm = [
            nn.BatchNorm1d(num_hidden, affine=False, track_running_stats=False)] * num_layers

    def encode(self, x):
        hidden1 = self.gc1(self.g, self.dropout(x))
        hidden1 = self.batch_norm[0](hidden1)
        return self.gc2(self.g, self.dropout(hidden1)), self.gc3(self.g, self.dropout(hidden1))

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # return self.dc(z), mu, logvar
        return z


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
