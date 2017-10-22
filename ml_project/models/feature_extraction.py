from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt


class ClusterExtraction(BaseEstimator, TransformerMixin):
    """Extract clusters to create features"""
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176)
        x_cut = X[:, :, 104, :]
        y_cut = X[:, 80, :, :]
        z_cut = X[:, :, :, 80]
        cuts = [x_cut, y_cut, z_cut]
        for cut in cuts:
            for sample in cut:
                connec = grid_to_graph(*sample.shape)
                ward = AgglomerativeClustering(
                    n_clusters=self.n_clusters,
                    linkage='ward', connectivity=connec)
                x = np.reshape(sample, (-1, 1))
                ward.fit(x)
                label = np.reshape(ward.labels_, sample.shape)
                self.showClusters(sample, label)

    def showClusters(self, sample, label):
        x = np.arange(sample.shape[0])
        y = np.arange(sample.shape[1])
        plt.figure()
        plt.imshow(sample, cmap=plt.cm.gray)
        plt.contourf(x, y, label, alpha=0.3, cmap=plt.cm.spectral)
        plt.show()
