from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression


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
    def __init__(self, threshold=50):
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

        return X_new

class KBestSelection(BaseEstimator, TransformerMixin):
    """"Select best features based on their variance"""
    def __init__(self, k=1000):
        self.k = k
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        self.components = SelectKBest(f_regression, k=self.k)
        self.components.fit(X, y)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        X_new = self.components.transform(X)

        return X_new
