import spacy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

# TODO: добавить возможность указывать путь до этого файла или подгрузить его из интернета
nlp = spacy.load('xx_ent_wiki_sm')


def preprocess(sentence):
    """ Preprocess sentence by changing all letters to lower case, replacing pronouns
    by ’-PRON-’, and removing all special characters except stop characters.

    :param sentence: source sentence

    :return clear sentence

    """
    clean_tokens = []
    sentence = nlp(sentence)
    for token in sentence:
        if not token.is_stop:
            clean_tokens.append(token.lemma_)
    return " ".join(clean_tokens)


def semhash_tokenizer(sentence, n=3):
    """ Convert sentence into semantic hash tokens.

    :param sentence: source sentence after preprocessing

    :return list of semantic hash tokens.

    """
    tokens = sentence.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [''.join(gram)
                         for gram in list(zip(*[list(hashed_token)[i:] for i in range(n)]))]
    return final_tokens


def semhash_corpus(corpus):
    """ Convert corpus into semantic hash corpus.

    :param corpus: list of unicode strings.

    :return list of semantic hash tokens.

    """
    new_corpus = []
    for sentence in corpus:
        sentence = preprocess(sentence)
        tokens = semhash_tokenizer(sentence)
        new_corpus.append(" ".join(map(str, tokens)))
    return new_corpus


class LogisticRegressionIntentClassifier(BaseEstimator, ClassifierMixin):
    """ Logistic regression classifier, which determines the user's intent. """
    def __init__(self):
        """ Create a new object """
        self.text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(penalty='l1', random_state=42)),
        ])

    def fit(self, X, y, **kwargs):
        """ Fit the seq2seq model to convert sequences one to another.

        :param X: input texts for training.
        :param y: target intents for training.

        :return self

        """
        X = semhash_corpus(X)
        self.text_clf.fit(X, y)
        return self

    def predict(self, X):
        """ Predict resulting intents by source sequences with a trained logistic regression model.
        Each sequence is unicode text composed from the tokens. Tokens are separated by spaces.

        :param X: source sequences.

        :return: resulting intents, predicted for source sequences.

        """
        X = semhash_corpus(X)
        return self.text_clf.predict(X)

    def fit_predict(self, X, y, **kwargs):
        return self.fit(X, y).predict(X)

