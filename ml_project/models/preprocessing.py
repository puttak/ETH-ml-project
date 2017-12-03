from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn import preprocessing
import numpy as np


class FFT(BaseEstimator, TransformerMixin):
    """
    Apply Fourier transform to time series
    """
    def __init__(self, save_path=None):
        self.save_path = save_path

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X_new = np.fft.fft(X)

        if self.save_path:
            np.save(self.save_path, X_new)

        return X_new
