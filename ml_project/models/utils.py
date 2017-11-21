import numpy as np
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold


def KMeansTransform(y, n_clusters):
    """
    Transform class probabilities into @n_clusters by using KMeans clustering
    and assigning labels for each cluster. Also return mean class probabilities
    for each cluster.
    """
    y_sorted = np.argsort(y)
    kmeans = KMeans(n_clusters=n_clusters, random_state=37).fit(y_sorted)
    y_new = kmeans.labels_
    clusters_proba = []
    for i in range(n_clusters):
        idx = (y_new == i)
        cluster = y[idx, :]
        mean_proba = np.mean(cluster, axis=0)
        clusters_proba.append(mean_proba)
    return y_new, clusters_proba


class MLStratifiedKFold(StratifiedKFold, TransformerMixin):
    """
    Wrapper around StratifiedKFold to allow use in multi-label classification
    """
    def __init__(self, n_clusters=6, n_splits=3,
                 shuffle=False, random_state=None):
        self.n_clusters = n_clusters
        super(MLStratifiedKFold, self).__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        y_new, clusters_proba = KMeansTransform(y, self.n_clusters)
        return super(MLStratifiedKFold, self).split(X, y_new)
