from sklearn.base import BaseEstimator, ClassifierMixin

from .LogisticRegressionIntentClassifier import LogisticRegressionIntentClassifier
from .MultiLayerPerceptronClassifier import MultiLayerPerceptronClassifier
from .USEIntentClassifier import USEIntentClassifier


class IntentClassifier(BaseEstimator, ClassifierMixin):
    """ IntentClassifier base class. """

    def __init__(self, method_name, *args, **kwargs):
        """ Create a new object with specified parameters.

        :param method_name: selected model

        """
        if method_name == 'log_reg':
            self.model = LogisticRegressionIntentClassifier(*args, **kwargs)
        elif method_name == 'perceptron':
            self.model = MultiLayerPerceptronClassifier(*args, **kwargs)
        elif method_name == 'use':
            self.model = USEIntentClassifier(*args, **kwargs)
        else:
            raise NotImplementedError(method_name)

    def fit(self, X, y, **kwargs):
        """ Fit the selected model to convert sequence to intent.

        :param X: input texts for training.
        :param y: target intents for training.

        :return self

        """
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        """ Predict resulting intents by source sequences with a trained selected model.
        
	    :param X: source sequences.

        :return: resulting intents, predicted for source sequences.

        """
        return self.model.predict(X)
