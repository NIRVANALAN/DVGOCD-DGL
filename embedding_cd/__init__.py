# from . import data
# from . import nn
# from . import metrics
# from . import sampler
# from . import train
# from . import utils
import time
from . import autoencoders
from . import node_embeddings

FACTORY = {
    'gcn': autoencoders.gcn,
    'gat': autoencoders.gat,
    'gae': autoencoders.GAE,
    'vgae': autoencoders.GVAE}


def create_model(name, g, **kwargs):
    if name not in FACTORY:
        raise NotImplementedError(f'{name} not in arch FACTORY')
    return FACTORY[name](g, **kwargs)


def embedding(embed_method, G, sage_model, feats):
    print("%%%%%% Starting Graph Embedding %%%%%%")
    if embed_method == "deepwalk":
        embed_start = time.process_time()
        embeddings = node_embeddings.deepwalk(G)

    elif embed_method == "node2vec":
        embed_start = time.process_time()
        embeddings = node_embeddings.node2vec(G)

    elif embed_method == "graphsage":
        # from .graphsage.graphsage import graphsage
        nx.set_node_attributes(G, False, "test")
        nx.set_node_attributes(G, False, "val")
        embed_start = time.process_time()
        embeddings = graphsage(G, feats, sage_model,
                               sage_weighted)
