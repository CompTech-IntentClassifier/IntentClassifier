#!/usr/bin/env python
# -*- coding: utf-8 -*-

import IntentClassifier

model = IntentClassifier()
sentence = "Жду ответа оператора"
cls = model.predict(sentence)
