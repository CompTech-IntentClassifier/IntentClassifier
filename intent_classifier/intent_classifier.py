# -*- coding: utf-8 -*-

"""

License: Apache License 2.0.

"""

from sklearn.base import BaseEstimator, ClassifierMixin

from .LogisticRegressionIntentClassifier import LogisticRegressionIntentClassifier


class IntentClassifier(BaseEstimator, ClassifierMixin):
    """ IntentClassifier base class. """

    def __init__(self, method_name):
        if method_name == 'log_reg':
            return LogisticRegressionIntentClassifier()

        raise NotImplementedError(method_name)

    def fit(self, X, y, **kwargs):
        """
        """
        return self

    def predict(self, X):
        """
        """
        return None

    def fit_predict(self, X, y, **kwargs):
        return None
