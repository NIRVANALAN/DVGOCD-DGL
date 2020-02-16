from .gat import GAT, GATConv
from .gcn import GCN

FACTORY = {
    'gcn' : GCN,
    'gat': GAT}

def create_model(name, g, **kwargs):
    if name not in FACTORY:
        raise NotImplementedError(f'{name} not in arch FACTORY')
    return FACTORY[name](g, **kwargs)