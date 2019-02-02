from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Sequential, load_model
from keras import layers
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.preprocessing.text import one_hot

# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd

from gensim.models import FastText, KeyedVectors

import os


# Pretrained data( dict.csv (word to index) and CNN_clf.h5 is in folder "CNN_model"

def f1_scores(output, y_true, class_list, aver='macro'):
    # function to count F1 score for
    y_pred = [np.argmax(output[i]) for i in range(len(output))]
    f1 = f1_score(y_true, y_pred, labels=class_list, average=aver)
    print(' *****  f1 = ', f1, ' **** ')
    return f1


def load_dict(path='CNN_model', fname='dict.csv'):
    dFrame = pd.read_csv(os.path.join(path, fname), sep='\t', index_col=False)
    words = dFrame['Unnamed: 0']
    indexes = dFrame['numbers']
    d = {indexes[i]: words[i] for i in range(len(indexes))}

    return d  # dictionary


def tokenize(sentence, dictionary):
    num_words = len(dictionary)
    tokenizer = Tokenizer(num_words,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ 1234567890', )
    tokenizer.word_index = dictionary
    max_words = 35
    # sentence_vec = text_to_word_sequence(sentence)
    sentence_vec = tokenizer.texts_to_sequences([sentence])
    print(sentence_vec)
    sentence_vec = sequence.pad_sequences(sentence_vec, maxlen=max_words)
    return sentence_vec


def tokenize(sentence, dictionary):
    num_words = len(dictionary)
    tokenizer = Tokenizer(num_words,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ 1234567890' )
    tokenizer.word_index = dictionary
    max_words = 35
    # sentence_vec = text_to_word_sequence(sentence)
    sentence_vec = tokenizer.texts_to_sequences([sentence])
    print(sentence_vec)
    sentence_vec = sequence.pad_sequences(sentence_vec, maxlen=max_words)
    return sentence_vec

def Preprocess_all(sentences):
    w = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 1234567890'
    for sentence in sentences:
        for wrond_char in w:  # Выпиливаем все знаки
            sentense = sentence.replace(wrond_char, ' ')

    return sentences


def check_input_data_to_fit(sentences, classes):
    if classes.dtype != 'int':
        classes_to_numbers(classes)


def ConvolutionalNN():
    return load_model("CNN_clf.h5")


class ConvolutionalNN_classifier:
    """"""

    def __init__(self):
        """"""
        self.text_clf = ConvolutionalNN()

        self.dict = load_dict()

    def fit(self, X, y, **kwargs):
        """ Fit the seq2seq model to convert sequences one to another.
        :param X: input texts for training.
        :param y: target intents for training.
        :return self
        """
        X = Preprocess_all(X, self.dict)
        self.text_clf.fit(X, y)
        # to do check of datatype (y)
        return self

    def predict(self, X):
        """ Predict resulting intents by source sequences with a trained logistic regression model.
        :param X: origin sequences.
        :return: resulting intents, predicted for source sequences.
        """
        X = Preprocess(X, self.dict)
        return np.argmax(self.text_clf.predict(X))
