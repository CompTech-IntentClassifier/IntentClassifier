import unittest
import numpy as np

from intent_classifier import IntentClassifier


class Tests(unittest.TestCase):

    def test_creation(self):
        IntentClassifier('log_reg')
        IntentClassifier('perceptron')

    def test_fit(self):
        logreg_clf = IntentClassifier('log_reg')
        perceptron_clf = IntentClassifier('perceptron')
        use_clf = IntentClassifier('use')

        for clf in [logreg_clf, perceptron_clf, use_clf]:
            try:
                clf.fit('aaa', [1])
                clf.fit(['aaa'], 1)
            except ValueError:
                pass
            else:
                raise ValueError

            clf.fit(['aaa'], [1])
            clf.fit(['aaa', 'sdfsd', 'sdfsdf'], [1, 2, 3])
            clf.fit(np.array(['aaa', 'sdfsd', 'sdfsdf']), np.array([1, 2, 3]))


if __name__ == '__main__':
    unittest.main()
