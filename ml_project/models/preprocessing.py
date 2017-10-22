from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn import preprocessing
import numpy as np

class Histogramize(BaseEstimator, TransformerMixin):
    def __init__(self, lcubes=9, nbins=100, spacing=10, scaling=False):
        self.lcubes = lcubes
        self.nbins = nbins
        self.spacing = spacing
        self.scaling = scaling

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176)
        X = X[:, self.spacing:-self.spacing, self.spacing:-self.spacing, self.spacing:-self.spacing]
        print("\n----------------------------------\n")
        print("Data shape after removing boundary: {}".format(X.shape))
        nx = int(X.shape[1]/self.lcubes)
        ny = int(X.shape[2]/self.lcubes)
        nz = int(X.shape[3]/self.lcubes)
        dx = X.shape[1] - nx*self.lcubes
        dy = X.shape[2] - ny*self.lcubes
        dz = X.shape[3] - nz*self.lcubes
        startx = int(np.floor(dx/2.0))
        starty = int(np.floor(dy/2.0))
        startz = int(np.floor(dz/2.0))
        stopx = -int(np.ceil(dx/2.0))
        stopy = -int(np.ceil(dy/2.0))
        stopz = -int(np.ceil(dz/2.0))
        print("Number of cubes for each axis: {}x{}x{} = {}".format(nx, ny, nz, nx*ny*nz))
        X = X[:, startx:stopx, starty:stopy, startz:stopz]
        print("Data shape after removing excess voxels: {}".format(X.shape))
        X_new = []
        for s in range(X.shape[0]):
            features = []
            a0 = np.array_split(X[s], nx, axis=0)
            n_cubes = 0
            for i in range(len(a0)):
                a1 = np.array_split(a0[i], ny, axis=1)
                for j in range(len(a1)):
                    a2 = np.array_split(a1[j], nz, axis=2)
                    for k in range(len(a2)):
                        cube_shape = np.array(a2[k].shape)
                        hist = np.histogram(a2[k], bins=self.nbins, range=(0, 3000))
                        features.append(hist[0])
                        n_cubes += 1
            features = np.array(features).flatten()
            X_new.append(features)
        X_new = np.array(X_new)
        if self.scaling:
            X_new = preprocessing.scale(X_new)
        print("Number of features: {}".format(X_new.shape[1]))
        return X_new


class Standardize(BaseEstimator, TransformerMixin):
    """Rescale data so that features have the properties of a normal distribution"""
    def __init__(self, norm='l1'):
        self.norm = norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176)
        x_cut = X[:, :, 104, :].reshape(X.shape[0], -1)
        y_cut = X[:, 80, :, :].reshape(X.shape[0], -1)
        z_cut = X[:, :, :, 80].reshape(X.shape[0], -1)
        stack = np.hstack([x_cut, y_cut, z_cut])
        idx = (np.std(stack, axis=0) == 0)
        X_standard = preprocessing.scale(stack[:, ~idx])
        X_standard = preprocessing.normalize(X_standard, norm=self.norm)
        return X_standard


class Flatten(BaseEstimator, TransformerMixin):
    """Flatten"""
    def __init__(self, dim=2):
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176) # Bad practice: hard-coded dimensions
        X = X.mean(axis=self.dim)
        return X_scaled.reshape(X.shape[0], -1)
