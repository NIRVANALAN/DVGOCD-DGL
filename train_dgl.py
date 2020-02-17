import argparse
import time
import pdb

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import load_data, register_data_args
from sklearn.preprocessing import normalize

import nocd
from dgl_cd.gcn import GCN
from dgl_cd.gat import GAT
from dgl_cd import create_model
from nocd.nn import BerpoDecoder

def evaluate(model_saver, model, features, Z_gt, thresh):
    model_saver.restore()
    model.eval()
    logits = F.relu(model(features))
    preds = logits.cpu().detach().numpy() > thresh
    nmi = nocd.metrics.overlapping_nmi(preds, Z_gt)
    # print(f'Final nmi = {nmi:.3f}')
    return nmi


def main(args):
    hidden_sizes = [128]    # hidden sizes of the GNN
    # weight_decay = 1e-5     # strength of L2 regularization on GNN weights
    # lr = 1e-3               # learning rate
    display_step = 50       # how often to compute validation loss
    balance_loss = True     # whether to use balanced loss
    stochastic_loss = True  # whether to use stochastic or full-batch training
    batch_size = 20000      # batch size (only for stochastic training)


    multitask_data = set(['ppi'])
    multitask = args.dataset in multitask_data

    # load and preprocess dataset
    if 'npz' in args.dataset:
        loader = nocd.data.load_dataset('data/mag_cs.npz')
        A, features, Z_gt = loader['A'], loader['X'], loader['Z']
        n_nodes, n_classes = Z_gt.shape
        graph = nx.Graph(A)
        feature_norm = normalize(features)
        features = torch.FloatTensor(feature_norm.todense())
        labels = torch.LongTensor(Z_gt)
        # x_norm = nocd.utils.to_sparse_tensor(feature_norm).cuda()
    else:
        data = load_data(args)
        train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
        train_feats = data.features[train_nid]
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_feats)
        features = scaler.transform(data.features)
        features = torch.FloatTensor(features)
        n_classes = data.num_labels
        if not multitask:
            labels = torch.LongTensor(data.labels)
        else:
            labels = torch.FloatTensor(data.labels)
        n_nodes = labels.shape[0]
        A = data.graph.adjacency_matrix(transpose=False)
        graph = data.graph
        pdb.set_trace()

    # add self loop
    if args.self_loop:
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph.add_edges_from(zip(graph.nodes(), graph.nodes()))
        print("adding self-loop edges")

    g = DGLGraph(graph, readonly=True)
    n_edges = g.number_of_edges()
    in_feats = features.shape[1]

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

    # create GAT model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = create_model(args.arch,g,
                num_layers=args.num_layers,
                in_dim=in_feats,
                num_hidden=args.num_hidden,
                num_classes=n_classes,
                heads=heads,
                activation=F.elu,
                feat_drop=args.in_drop,
                attn_drop = args.attn_drop,
                negative_slope=args.negative_slope,
                residual=args.residual)


    if cuda:
        model.cuda()
    print(model)
    sampler = nocd.sampler.get_edge_sampler(A, batch_size, batch_size, num_workers=5)
    decoder = nocd.nn.BerpoDecoder(n_nodes, A.nnz, balance_loss=balance_loss) # ? nnz: number of nonzero values
    # use optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,)
                                #  weight_decay=args.weight_decay)
    # initialize graph
    ##########################################
    val_loss = np.inf
    validation_fn = lambda: val_loss
    early_stopping = nocd.train.NoImprovementStopping(validation_fn, patience=5)
    model_saver = nocd.train.ModelSaver(model)

    for epoch, batch in enumerate(sampler):
        if epoch > args.n_epochs:
            break
        if epoch % display_step == 0:
            with torch.no_grad():
                model.eval()
                # Compute validation loss
                logits = F.relu(model(features))
                val_loss = decoder.loss_full(logits, A)
                
                # Check if it's time for early stopping / to save the model
                early_stopping.next_step()
                if early_stopping.should_save():
                    model_saver.save()
                if early_stopping.should_stop():
                    print(f'Breaking due to early stopping at epoch {epoch}')
                    break
                print(f'Epoch {epoch:4d}, loss.full = {val_loss:.4f}, NMI = {evaluate(model_saver,model, features,Z_gt, thresh=args.thresh):.4f}')
                
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = F.relu(model(features))
        ones_idx, zeros_idx = batch
        if stochastic_loss:
            loss = decoder.loss_batch(logits, ones_idx, zeros_idx)
        else:
            loss = decoder.loss_full(logits, A)
        loss += nocd.utils.l2_reg_loss(model, scale=args.weight_decay)
        loss.backward()
        optimizer.step()
    print(f'Final NMI = {evaluate(model_saver,model, features,Z_gt, args.thresh):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=5e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=2000,
            help="number of training epochs")
    # parser.add_argument("--n-hidden", type=int, default=128,
    #         help="number of hidden gcn units")
    # parser.add_argument("--n-layers", type=int, default=1,
    #         help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--thresh", type=float, default=0.5,
            help="Threshold for Affilicaiton matrix")
    parser.add_argument("--self-loop", action='store_false',
            help="graph self-loop (default=True)")
    # parser.set_defaults(self_loop=False)
    # GAT args
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    # MODEL
    parser.add_argument("--arch", type=str, default='gcn', help='the arch of gcn model')

    args = parser.parse_args()

    print(args)
    main(args)
