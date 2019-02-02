import unittest
import numpy as np

from intent_classifier import IntentClassifier


class Tests(unittest.TestCase):

    def test_creation(self):
        IntentClassifier('log_reg')

    def test_fit(self):
        clf = IntentClassifier('log_reg')

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
