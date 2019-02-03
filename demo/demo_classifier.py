#!/usr/bin/env python
# -*- coding: utf-8 -*-

from intent_classifier import IntentClassifier

model = IntentClassifier()
sentence = "Жду ответа оператора"
cls = model.predict(sentence)
