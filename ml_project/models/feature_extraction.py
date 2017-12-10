from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg


class RDistanceExtractor(BaseEstimator, TransformerMixin):
    """
    Extract R-peaks from signal using a segmenter from the biosppy package,
    divide the signal in n bin and compute the mean and variance of
    the distance between two peaks in each bin.

    Args:
        - nbins (integer): number of bins into which the signal is divided
        - segmenter (string): name of segmenter from biosppy package
                              to be used
        - sampling_rate (float): sampling rate
        - save_path (string): path to where the new features will be saved
        - settings: extra parameters to pass to the segmenter
    """
    def __init__(self, nbins=20, segmenter='christov',
                 sampling_rate=200.0, save_path=None, settings=None):
        self.nbins = nbins
        self.segmenter_name = segmenter
        self.segmenter = getattr(ecg, segmenter+'_segmenter')
        self.sampling_rate = sampling_rate
        self.save_path = save_path
        if settings:
            self.settings = settings
        else:
            self.settings = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("Extracting features with {} segmenter"
               .format(self.segmenter_name))
        N = X.shape[0]
        X_new = np.zeros((N, self.nbins, 2))
        sbin = X.shape[1]/self.nbins
        for i in range(N):
            output = self.segmenter(signal=X[i, :],
                            sampling_rate=self.sampling_rate, **self.settings)
            rpeaks = output['rpeaks']
            distances = np.diff(rpeaks)
            max_idx = len(distances)
            for j in range(self.nbins):
                window = [j*sbin, (j+1)*sbin]
                selection = np.where(np.logical_and(rpeaks>window[0],
                                     rpeaks<window[1]))[0]
                if max_idx in selection:
                    current_distances = distances[selection[:-1]]
                else:
                    current_distances = distances[selection]
                if len(current_distances) > 0:
                    current_mean = np.mean(current_distances)
                    current_var = np.var(current_distances)
                else:
                    current_mean = -1
                    current_var = -1
                X_new[i, j, :] = [current_mean, current_var]
            if (i+1) % 100 == 0:
                print("- progress: {}/{}".format(i+1, N))
        X_new = X_new.reshape((N, self.nbins*2))
        if self.save_path is not None:
            np.save(self.save_path, X_new)
        return X_new
