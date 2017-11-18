import numpy as np
from sklearn.cluster import KMeans


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
