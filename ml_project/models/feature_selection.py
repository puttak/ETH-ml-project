from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


class RandomForestSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators=20, n_jobs=4,
                 class_weight='balanced_subsample', threshold='mean',
                 verbose=False):
        self.clf = RandomForestClassifier(n_estimators=n_estimators,
                                          n_jobs=n_jobs,
                                          class_weight=class_weight)
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y=None):
        X = check_array(X)
        self.model = SelectFromModel(self.clf,
                                     threshold=self.threshold).fit(X, y)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self.model, ["estimator_"])
        X = check_array(X)
        X_new = self.model.transform(X)
        if self.verbose:
            print("Selected {} features from {}"
                  .format(X_new.shape[1], X.shape[1]))
        return X_new
