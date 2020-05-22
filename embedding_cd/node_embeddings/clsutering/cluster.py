from .kmeans_torch import kmeans as kmeans_torch
from .lloyd.kmeans import lloyd

cluster_methods = {
    'kmeans': kmeans_torch,
    # 'kmeans_torch': kmeans_torch,
}  # density peak, DBSCAN,


def cluster(X, num_clusters, distance='cosine', method='kmeans') -> tuple:
    """ Embedding Custer
    return cluster_id, cluster_centroid
    """
    assert method in cluster_methods
    cluster_ids_x, cluster_centers = cluster_methods[method](
        X, num_clusters, distance=distance
    )
    return cluster_ids_x, cluster_centers


__all__ = ['cluster']  # ! from cluster import *