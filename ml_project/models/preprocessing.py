from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn import preprocessing
import numpy as np

class Standardize(BaseEstimator, TransformerMixin):
    """Rescale data so that features have the properties of a normal distribution"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176)
        x_cut = X[:, :, 104, :].reshape(X.shape[0], -1)
        y_cut = X[:, 80, :, :].reshape(X.shape[0], -1)
        z_cut = X[:, :, :, 80].reshape(X.shape[0], -1)
        stack = np.hstack([x_cut, y_cut, z_cut])
        return preprocessing.scale(stack)


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
