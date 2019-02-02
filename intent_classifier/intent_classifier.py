# -*- coding: utf-8 -*-

"""

License: Apache License 2.0.

"""

from sklearn.base import BaseEstimator, ClassifierMixin

from .LogisticRegressionIntentClassifier import LogisticRegressionIntentClassifier


class IntentClassifier(BaseEstimator, ClassifierMixin):
    """ IntentClassifier base class. """
    def __init__(self, method_name):
	""" Create a new object with specified parameters.
		
	:param method_name: selected model

	"""
        if method_name == 'log_reg':
            self.model = LogisticRegressionIntentClassifier()
        else:
            raise NotImplementedError(method_name)

    def fit(self, X, y, **kwargs):
        """ Fit the selected model to convert sequence to intent.

        :param X: input texts for training.
        :param y: target intents for training.

        :return self

        """
        return self.model.fit(X, y)

    def predict(self, X):
        """ Predict resulting intents by source sequences with a trained selected model.
        
	:param X: source sequences.

        :return: resulting intents, predicted for source sequences.

        """
        return self.model.predict(X)

    def fit_predict(self, X, y, **kwargs):
        return self.model.fit_predict(X,y)
