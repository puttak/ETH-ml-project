from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from ml_project.models.utils import KMeansTransform


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


class MLPercentileSelection(BaseEstimator, TransformerMixin):
    """"Select best features based on their variance"""
    def __init__(self, percentile=10, n_clusters=6):
        self.percentile = percentile
        self.n_clusters = n_clusters
        self.mask = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        y_new, clusters_proba = KMeansTransform(y, self.n_clusters)
        selector = SelectPercentile(f_classif, percentile=self.percentile)
        selector.fit(X, y_new)
        self.mask = selector.get_support()
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["mask"])
        X = check_array(X)
        X_new = X[:, self.mask]
        print("Number of features after PercentileSelection(percentile={}): {}"
              .format(self.percentile, X_new.shape[1]))
        return X_new
