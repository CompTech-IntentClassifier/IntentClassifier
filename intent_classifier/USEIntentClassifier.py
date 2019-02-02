import xgboost as xgb
import tensorflow as tf
import tensorflow_hub as hub

from googletrans import Translator
from sklearn.base import BaseEstimator, ClassifierMixin


class USEIntentClassifier(BaseEstimator, ClassifierMixin):
    """ Logistic regression classifier, which determines the user's intent. """

    def __init__(self, module_url='https://tfhub.dev/google/universal-sentence-encoder/2', translate_to_en=True):
        """ Create a new object """
        # Import the Universal Sentence Encoder's TF Hub module
        self.embed = hub.Module(module_url)
        tf.logging.set_verbosity(tf.logging.ERROR)

        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        # XGBoost
        self.model_xgb = xgb.XGBClassifier()

        # Use
        self.translate_to_en = translate_to_en

        # Google Translate
        if translate_to_en:
            self.translator = Translator()

    def fit(self, X, y, **kwargs):
        """ Fit the logistic regression model to convert sequence to intent.

        :param X: input texts for training.
        :param y: target intents for training.

        :return self

        """
        X = self.__tokenize(X)
        self.model_xgb.fit(X, y)
        return self

    def predict(self, X):
        """ Predict resulting intents by source sequences with a trained logistic regression model.

        :param X: source sequences.

        :return: resulting intents, predicted for source sequences.

        """
        X = self.__tokenize(X)
        return self.model_xgb.predict(X)

    def __tokenize(self, X):
        if self.translate_to_en:
            X = self.__translate_text(X)
        return self.__get_embeds(X)

    def __get_embeds(self, messages):
        return self.session.run(self.embed(messages))

    def __translate_text(self, texts):
        translations = self.translator.translate(texts, dest='en', src='ru')
        return [translation.text for translation in translations]
