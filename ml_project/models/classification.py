import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class RandomForestPredictor(RandomForestClassifier, TransformerMixin):
    """
    Classification with randomized decision trees
    """
    def __init__(self, n_estimators=20, n_jobs=4,
                 class_weight='balanced_subsample', debug=False):
        self.debug = debug
        super(RandomForestPredictor, self).__init__(
            n_estimators=n_estimators, n_jobs=n_jobs, class_weight=class_weight)

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        super(RandomForestPredictor, self).fit(X, y)
        if self.debug:
            print("Fitting {} samples:".format(X.shape[0]))
            print("\t- feature importance: {}".format(self.feature_importances_))
        return self

    def predict(self, X):
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)
        prediction = super(RandomForestPredictor, self).predict(X)
        if self.debug:
            print("Predicting {} samples".format(X.shape[0]))
        return int(prediction)

    def score(self, X, y):
        prediction = self.predict(X)
        score = f1_score(prediction, y, average='micro')
        if self.debug:
            print("Score for {} samples: {}".format(X.shape[0], score))
        return score
