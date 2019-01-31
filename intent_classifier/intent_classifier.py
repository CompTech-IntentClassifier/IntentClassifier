# -*- coding: utf-8 -*-

"""

License: Apache License 2.0.

"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class IntentClassifier(BaseEstimator, ClassifierMixin):
    """ Sequence-to-sequence classifier, which converts one language sequence into another. """

    def __init__(self):
        """ Create a new object with specified parameters.

        """
        pass

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

    def load_weights(self, weights_as_bytes):
        """
        """
        pass

    def dump_weights(self):
        """
        """
        pass

    def get_params(self, deep=True):
        """
        """
        pass

    def set_params(self, **params):
        """
        """
        pass

    def dump_all(self):
        """
        """
        pass

    def load_all(self, new_params):
        """
        """
        pass

    @staticmethod
    def check_params(**kwargs):
        """
        """
        pass

    @staticmethod
    def check_X(X, checked_object_name=u'X'):
        """
        """
        pass

    @staticmethod
    def tokenize_text(src, lowercase):
        """
        """
        return list(filter(lambda it: len(it) > 0, src.strip().lower().split() if lowercase else src.strip().split()))
