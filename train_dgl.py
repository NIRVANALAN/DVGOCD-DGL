import argparse
import time
import pdb

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data, register_data_args
from sklearn.preprocessing import normalize

import nocd
from dgl_cd.gcn import GCN, mp_GCN, spmv_GCN
from nocd.nn import BerpoDecoder


def main(args):
    hidden_sizes = [128]    # hidden sizes of the GNN
    # weight_decay = 1e-5     # strength of L2 regularization on GNN weights
    # lr = 1e-3               # learning rate
    # max_epochs = 500        # number of epochs to train
    display_step = 25       # how often to compute validation loss
    balance_loss = True     # whether to use balanced loss
    stochastic_loss = True  # whether to use stochastic or full-batch training
    batch_size = 20000      # batch size (only for stochastic training)
    thresh = args.thresh
    # load and preprocess dataset
    # data = load_data(args)
    loader = nocd.data.load_dataset('data/mag_cs.npz')
    A, features, Z_gt = loader['A'], loader['X'], loader['Z']
    n_nodes, n_classes = Z_gt.shape
    graph = nx.Graph(A)
    feature_norm = normalize(features)
    features = torch.FloatTensor(feature_norm.todense())
    labels = torch.LongTensor(Z_gt)
    in_feats = features.shape[1]
    n_edges = graph.number_of_edges()
    x_norm = nocd.utils.to_sparse_tensor(feature_norm).cuda()
    # gnn = nocd.nn.ImprovedGCN(x_norm.shape[1], hidden_sizes, n_classes).cuda()
    # adj_norm = gnn.normalize_adj(A)
    print("""----Data statistics------'
      #Edges %d
      #Classes %d """
      %(n_edges, n_classes))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()

    # graph preprocess and calculate normalization factor
    # add self loop
    if args.self_loop:
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph.add_edges_from(zip(graph.nodes(), graph.nodes()))
    g = DGLGraph(graph)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout,
                batch_norm=True)

    if cuda:
        model.cuda()
    print(model)
    sampler = nocd.sampler.get_edge_sampler(A, batch_size, batch_size, num_workers=5)
    decoder = nocd.nn.BerpoDecoder(n_nodes, A.nnz, balance_loss=balance_loss) # ? nnz: number of nonzero values
    # use optimizer
    # model = gnn
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,)
                                #  weight_decay=args.weight_decay)
    # initialize graph
    ##########################################
    val_loss = np.inf
    validation_fn = lambda: val_loss
    early_stopping = nocd.train.NoImprovementStopping(validation_fn, patience=10)
    model_saver = nocd.train.ModelSaver(model)

    for epoch, batch in enumerate(sampler):
        if epoch > args.n_epochs:
            break
        if epoch % 25 == 0:
            with torch.no_grad():
                model.eval()
                # Compute validation loss
                # logits = F.relu(model(x_norm, adj_norm))
                logits = F.relu(model(features))
                val_loss = decoder.loss_full(logits, A)
                print(f'Epoch {epoch:4d}, loss.full = {val_loss:.4f}')
                
                # Check if it's time for early stopping / to save the model
                early_stopping.next_step()
                if early_stopping.should_save():
                    model_saver.save()
                if early_stopping.should_stop():
                    print(f'Breaking due to early stopping at epoch {epoch}')
                    break
                
        # Training step
        model.train()
        optimizer.zero_grad()
        # logits = F.relu(model(x_norm, adj_norm))
        logits = F.relu(model(features))
        ones_idx, zeros_idx = batch
        if stochastic_loss:
            loss = decoder.loss_batch(logits, ones_idx, zeros_idx)
        else:
            loss = decoder.loss_full(logits, A)
        loss += nocd.utils.l2_reg_loss(model, scale=args.weight_decay)
        loss.backward()
        optimizer.step()

    model_saver.restore()
    model.eval()
    # logits = F.relu(model(x_norm, adj_norm))
    logits = F.relu(model(features))
    preds = logits.cpu().detach().numpy() > thresh
    nmi = nocd.metrics.overlapping_nmi(preds, Z_gt)
    print(f'Final nmi = {nmi:.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=500,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
            help="Weight for L2 loss")
    parser.add_argument("--thresh", type=float, default=0.5,
            help="Threshold for Affilicaiton matrix")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
