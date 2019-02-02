# IntentClassifier

This is IntentClassifier, a library on Python implemented classification of intents in short sentenses. The aim of library is to realize the effective intent detection specified for the Russian language.

The library conrains the next clssification algorithms' implementations:
* Logistic Regression + Subword Semantic Hashing
* Bert + LogReg
* USE
* LSTM + Attention
* CNN
* Mpclassifier+Tf-IDF

Report on the results of training and testing can be found in the "report"
Short information about classes is given in the "Algorithms". 
To get more information about used algorithms and all the Deep Learning field some additional information sources are presented at Knowledge Bank

Advantages of library in comparison with … is the next
* classify intents to 10 classes
* high quality (F score( macro) obtained by small dataset (< 1000 sentenses) via cross-validation.

This is pretrained on real dataset of peoples’ intents to chat-bot with mobile operator library.


## Getting Started

### Installing

To install this project on your local machine, you should run the following commands in Terminal:

```
git clone https://github.com/CompTech-IntentClassifier/IntentClassifier.git
cd IntentClassifier
sudo python setup.py
```

You can also run the tests

```
python setup.py test
```

But I recommend you to use pip and install this package from PyPi:

```
pip install comp_tech_intent_classifier
```

or

```
sudo pip install comp_tech_intent_classifier
```

### Usage

TODO
