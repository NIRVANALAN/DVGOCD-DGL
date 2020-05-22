# from . import data
# from . import nn
# from . import metrics
# from . import sampler
# from . import train
# from . import utils
from . import autoencoders
from . import node_embeddings

FACTORY = {
    'gcn' : autoencoders.gcn,
    'gat': autoencoders.gat,
    'gae': autoencoders.GAE,
    'vgae': autoencoders.GVAE}


def create_model(name, g, **kwargs):
    if name not in FACTORY:
        raise NotImplementedError(f'{name} not in arch FACTORY')
    return FACTORY[name](g, **kwargs)
