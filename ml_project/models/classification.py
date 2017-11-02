import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.stats import spearmanr


class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""
    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))

    def score(self, X, y):
        a = self.predict_proba(X)
        rhos = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            rhos[i] = spearmanr(a[i, :], y[i, :], axis=0)[0]
        score = rhos.mean()
        return score
