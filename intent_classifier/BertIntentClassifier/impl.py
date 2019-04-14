import spacy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression



##########################


import codecs
import collections
import json
import re
import numpy as np
import modeling
import tokenization
import tensorflow as tf

import pandas as pd

import tokenization

import sys

from extract_features import InputExample, InputFeatures, input_fn_builder, model_fn_builder

from extract_features import convert_examples_to_features, _truncate_seq_pair, read_examples

tf.logging.set_verbosity(tf.logging.ERROR)


BERT_BASE_DIR = '/Users/kolsha/Documents/Projects/Python/BERT/multi_cased_L-12_H-768_A-12'

init_checkpoint = BERT_BASE_DIR + '/bert_model.ckpt'

layer_indexes = [-1]

use_one_hot_embeddings = False

max_seq_length = 128

bert_config = modeling.BertConfig.from_json_file(BERT_BASE_DIR +'/bert_config.json')

tokenizer = tokenization.FullTokenizer(
      vocab_file=BERT_BASE_DIR+ '/vocab.txt', do_lower_case=False)

for (j, layer_index) in enumerate(layer_indexes):
    print(j, layer_index)


is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
  master=None,
  tpu_config=tf.contrib.tpu.TPUConfig(
      per_host_input_for_training=is_per_host)
)

def convert_lines_to_examples(lines):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    for line in lines:
        line = tokenization.convert_to_unicode(line)
        if not line:
            continue
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        examples.append(
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
        unique_id += 1
    return examples
##########################



class BertIntentClassifier(BaseEstimator, ClassifierMixin):
    """  """

    def __init__(self, spacy_load_path='xx_ent_wiki_sm'):
        """ Create a new object """
        self.nlp = spacy.load(spacy_load_path)
        self.text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(penalty='l1', random_state=42)),
        ])

    def fit(self, X, y, **kwargs):
        """ Fit the logistic regression model to convert sequence to intent.

        :param X: input texts for training.
        :param y: target intents for training.

        :return self

        """
        X = self.__semhash_corpus(X)
        self.text_clf.fit(X, y)
        return self

    def predict(self, X):
        """ Predict resulting intents by source sequences with a trained logistic regression model.

        :param X: source sequences.

        :return: resulting intents, predicted for source sequences.

        """
        X = self.__semhash_corpus(X)
        return self.text_clf.predict(X)

    def __preprocess(self, sentence):
        """ Preprocess sentence by changing all letters to lower case, replacing pronouns
        by ’-PRON-’, and removing all special characters except stop characters.

        :param sentence: origin sentence as list of sentense of String type

        :return clear sentence as list  of sentense of String type

        """
        clean_tokens = []
        sentence = self.nlp(sentence)
        for token in sentence:
            if not token.is_stop:
                clean_tokens.append(token.lemma_)
        return " ".join(clean_tokens)

    def __semhash_tokenizer(self, sentence, n=3):
        """ Convert sentence into semantic hash tokens.

        :param sentence: origin sentence after preprocessing  as 1D list of sentense of string type

        :return list of semantic hash tokens as np.array, ndim = 2

        """
        tokens = sentence.split(" ")
        final_tokens = []
        for unhashed_token in tokens:
            hashed_token = "#{}#".format(unhashed_token)
            final_tokens += [''.join(gram)
                             for gram in list(zip(*[list(hashed_token)[i:] for i in range(n)]))]
        return final_tokens

    def __semhash_corpus(self, corpus):
        """ Convert corpus into semantic hash corpus.

        :param corpus: list of unicode strings.

        :return list of semantic hash tokens.

        """
        new_corpus = []
        for sentence in corpus:
            sentence = self.__preprocess(sentence)
            tokens = self.__semhash_tokenizer(sentence)
            new_corpus.append(" ".join(map(str, tokens)))
        return new_corpus
