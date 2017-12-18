from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from biosppy.signals import ecg
from time import time
from os.path import join


class RDistanceExtractor(BaseEstimator, TransformerMixin):
    """
    Extract R-peaks from signal using a segmenter from the biosppy package,
    divide the signal in n bin and compute the mean, the variance and the
    histogram of the distance between two peaks in each bin.

    Args:
        - n_bins (integer): number of bins into which the signal is divided
        - n_hist (integer): number of bins to use in histogram
        - segmenter (string): name of segmenter from biosppy package
                              to be used
        - sampling_rate (float): sampling rate
        - save_path (string): path to where the new features will be saved
        - settings (obj): extra parameters to pass to the segmenter
        - n_jobs (int): number of processes to use in parallel
        - verbose (bool): wheter to output debugging text or not
    """
    def __init__(self, n_bins=20, n_hist=10, segmenter='christov',
                 sampling_rate=200.0, save_path=None,
                 settings=None, n_jobs=4, verbose=False):
        self.n_bins = n_bins
        self.n_hist = n_hist
        self.segmenter = getattr(ecg, segmenter+'_segmenter')
        self.sampling_rate = sampling_rate
        self.save_path = save_path
        self.n_jobs = n_jobs
        self.verbose = verbose
        if settings:
            self.settings = settings
        else:
            self.settings = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        N = X.shape[0]
        self.X_new = np.zeros((N, self.n_bins*2+self.n_hist))
        pool = mp.Pool(processes=self.n_jobs)
        if self.verbose:
            start = time()
            print("Extracting features with {} segmenter"
                    .format(self.segmenter.__name__))
        for i in range(N):
            pool.apply_async(self._compute_features,
                args=(i, X[i, :]), callback=self._log_result)
        pool.close()
        pool.join()
        if self.verbose:
            end = time()
            minutes, seconds = divmod(end-start, 60)
            print("Finished feature extraction in {:0>2}:{:05.2f}s"
                  .format(int(minutes), seconds))
        if self.save_path:
            fname = "features_{}_{}.npy".format(
                    self.segmenter.__name__, int(self.sampling_rate))
            np.save(join(self.save_path, fname), self.X_new)
        return self.X_new

    def _compute_features(self, index, signal):
        if self.verbose:
            start = time()
        features = np.zeros(self.n_bins*2+self.n_hist)
        bin_size = len(signal)/self.n_bins
        bin_stats = np.zeros((self.n_bins, 2))
        rpeaks = self.segmenter(signal=signal, sampling_rate=self.sampling_rate,
                           **self.settings)['rpeaks']
        distances = np.diff(rpeaks)
        if len(distances) > 0:
            max_idx = len(distances)
            for i in range(self.n_bins):
                window = [i*bin_size, (i+1)*bin_size]
                selection = np.where(np.logical_and(rpeaks>window[0], rpeaks<window[1]))[0]
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
                bin_stats[i, :] = [current_mean, current_var]
            hist, _ = np.histogram(distances, bins=self.n_hist, density=True)
            features[:self.n_bins*2] = bin_stats.flatten()
            features[-self.n_hist:] = hist
        result = {
            'index': index,
            'features': features}
        if self.verbose:
            end = time()
            result['exec_time'] = end-start
        return result

    def _log_result(self, res):
        self.X_new[res['index']] = res['features']
        minutes, seconds = divmod(res['exec_time'], 60)
        if self.verbose:
            print("\t- sample #{}: done in {:0>2}:{:05.2f}s"
                  .format(res['index'], int(minutes), seconds))
