import numpy as np
import spacy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# TODO:
nlp = spacy.load('xx_ent_wiki_sm')


def tokenize(doc):
    tokens = []
    doc = nlp.tokenizer(doc)
    for token in doc:
        tokens.append(token.text)
    return tokens


def preprocess(doc):
    clean_tokens = []
    doc = nlp(doc)
    for token in doc:
        if not token.is_stop:
            clean_tokens.append(token.lemma_)
    return " ".join(clean_tokens)


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def semhash_tokenizer(text):
    tokens = text.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [''.join(gram)
                         for gram in list(find_ngrams(list(hashed_token), 3))]
    return final_tokens


def semhash_corpus(corpus):
    new_corpus = []
    for sentence in corpus:
        sentence = preprocess(sentence)
        tokens = semhash_tokenizer(sentence)
        new_corpus.append(" ".join(map(str, tokens)))
    return new_corpus


class NaiveBayesIntentClassifier(BaseEstimator, ClassifierMixin):
    """
    """

    def __init__(self):
        """
        """
        self.text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(penalty='l1', random_state=42)),
        ])

    def fit(self, X, y, **kwargs):
        """
        """
        self.text_clf.fit(X, y)
        return self

    def predict(self, X):
        """
        """
        return self.text_clf.predict(X)

    def fit_predict(self, X, y, **kwargs):
        return self.fit(X, y).predict(X)
