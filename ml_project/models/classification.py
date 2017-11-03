import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LinearRegression, LogisticRegression
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

class MLLinearPredictor(LinearRegression, TransformerMixin):
    """
    Perform linear regression on the data for every target label independently
    """
    def __init__(self):
        super(MLLinearPredictor, self).__init__()

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        y = np.tan((y - 0.5)*np.pi)
        super(MLLinearPredictor, self).fit(X, y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X)
        prediction = super(MLLinearPredictor, self).predict(X)
        prediction = np.arctan(prediction)*1.0/np.pi + 0.5
        for i in range(prediction.shape[0]):
            prediction[i, :] = prediction[i, :] / sum(prediction[i, :])
        return prediction

    def score(self, X, y):
        a = self.predict_proba(X)
        rhos = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            rhos[i] = spearmanr(a[i, :], y[i, :], axis=0)[0]
        score = rhos.mean()
        return score


class MLLogisticRegression(LogisticRegression, TransformerMixin):
    """
    Perform multinomial logistic regression on dataset
    """
    def __init__(self):
        super(MLLogisticRegression, self).__init__(solver='sag',
                                                   multi_class='multinomial')

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        super(MLLogisticRegression, self).fit(X, y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X)
        prediction = super(MLLogisticRegression, self).predict_proba(X)
        for i in range(prediction.shape[0]):
            prediction[i, :] = prediction[i, :] / sum(prediction[i, :])
        return prediction

    def score(self, X, y):
        a = self.predict_proba(X)
        rhos = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            rhos[i] = spearmanr(a[i, :], y[i, :], axis=0)[0]
        score = rhos.mean()
        return score
