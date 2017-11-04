import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
from sklearn.feature_selection import VarianceThreshold, f_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile


class NonZeroSelection(BaseEstimator, TransformerMixin):
    """Select non-zero voxels"""
    def fit(self, X, y=None):
        X = check_array(X)
        self.nonzero = X.sum(axis=0) > 0

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["nonzero"])
        X = check_array(X)
        return X[:, self.nonzero]


class RandomSelection(BaseEstimator, TransformerMixin):
    """Random Selection of features"""
    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
                            n_features,
                            self.n_components,
                            random_state=random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new


class VarianceSelection(BaseEstimator, TransformerMixin):
    """"Select best features based on their variance"""
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        self.components = VarianceThreshold(self.threshold)
        self.components.fit(X)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        X_new = self.components.transform(X)
        print("Number of features after VarianceSelection(threshold={}): {}"
              .format(self.threshold, X_new.shape[1]))
        return X_new


class KBestSelection(BaseEstimator, TransformerMixin):
    """"Select best features based on their variance"""
    def __init__(self, k=1000):
        self.k = k
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        if self.k > X.shape[1]:
            self.components = SelectKBest(f_regression, k='all')
            print("WARNING: KBestSelection used k='all'")
        else:
            self.components = SelectKBest(f_regression, k=self.k)
        self.components.fit(X, y)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        X_new = self.components.transform(X)
        print("Number of features after KBestSelection(k={}): {}"
              .format(self.k, X_new.shape[1]))
        return X_new


class PercentileSelection(BaseEstimator, TransformerMixin):
    """"Select best features based on their variance"""
    def __init__(self, percentile=10):
        self.percentile = percentile
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        self.components = SelectPercentile(f_regression,
                                           percentile=self.percentile)
        self.components.fit(X, y)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        X_new = self.components.transform(X)
        print("Number of features after PercentileSelection(percentile={}): {}"
              .format(self.percentile, X_new.shape[1]))
        return X_new


class MLPercentileSelection(BaseEstimator, TransformerMixin):
    """"Select best features based on their variance"""
    def __init__(self, percentile=10):
        self.percentile = percentile
        self.mask = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        mask = np.zeros(X.shape[1], dtype=bool)
        for i in range(y.shape[1]):
            selector = SelectPercentile(f_regression,
                                           percentile=self.percentile)
            selector.fit(X, y[:, i])
            if i == 0:
                mask += selector.get_support()
            else:
                mask = np.logical_and(mask, selector.get_support())
        self.mask = mask
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["mask"])
        X = check_array(X)
        X_new = X[:, self.mask]
        print("Number of features after PercentileSelection(percentile={}): {}"
              .format(self.percentile, X_new.shape[1]))
        return X_new
