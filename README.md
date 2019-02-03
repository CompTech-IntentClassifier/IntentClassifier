# IntentClassifier
Это библиотека предобученных алгоритмов для классификации намерений человека при общении с чат-ботом. Библиотека была разработана в рамках проекта на зимней школе CompTech@NSK в Новосибирске в 2019 году. Идея проекта была сформулирована компанией DataMonsters.
 Задачи библиотеки:
 * выполнять классификацию клиентских намерений (intents) в текстах на русском языке с высокой точностью и за малое время;
 * выполнять дообучение на новом датасете;
 

## Возможности для использования:
Предобученную модель можно использовать в качестве компонента чат-бота мобильных операторов при общении с клиентом. 
Разработанные в рамках текущего проекта модели можно дообучать на новых данных для решения следующих бизнес-задач:
* реализация чат-ботов на любую тематику (заказ покупок на дом, консультация по выдаче кредитов, общение с клиентом для возврата просроченной задолженности и т.п.);
* классификация заголовков газет, рекламы;
* автоматическая сортировка сообщений по темам в почтовом ящике;
* тематический анализ в онлайн-советчике при выборе фильма, книги, еды, места и т.п;
* и прочие.

Текущая версия библиотеки легла в основу демонстрационного чат-бота, разработанного на базе платформы ВКонтакте для общения с клиентом на тему услуг, предоставляемых мобильными операторами. Размеченные данные для обучения модели предоставлены компанией Data Monsters и включают реальные переговоры пользователей из общения с чат-ботом на основе фреймворка [Electra](https://electraai.dev.datamonsters.com).

 ## Структура библиотеки
 В библиотеке представлены следующие алгоритмы машинного обучения. 
* [Bert](https://github.com/google-research/bert) + Логистическая регрессия
* Сверточная нейронная сеть + FastText проекта [RusVectores](https://rusvectores.org/ru/)
* Universal Sentence Encoder ([USE](https://tfhub.dev/google/universal-sentence-encoder/2)) + Google Translate + градиентный бустинг (XGBoostClassifier) 
* Tf-IDF + Классический персептрон
* Subword Semantic Hashing + Логистическая регрессия
* двунаправленная рекуррентная сеть типа LSTM + MaxPooling + FastText проекта [RusVectores](https://rusvectores.org/ru/)

Модель была разработана специально для работы с обучающими датасетами небольшого объёма.
Каждый из алгоритмов был предобучен на датасете, состоящем из русскоязычных обращений клиентов к чат-боту мобильного оператора. В датасете представлено 877 текстовых обращений клиентов. разбитых вручную на 10 классов намерений.

Оценка качества алгоритмов проводилось с использование кросс-валидации на 10 блоках. Подробные результаты тестирования с описанием алгоритмов представлены в документе [Test Results](https://github.com/CompTech-IntentClassifier/IntentClassifier/blob/master/Test%20Results.md). Все реализованные алгоритмы показали хорошее качество классификации выше 89% (критерием качества являлась сбалансированная F-мера).

## Возможности библиотеки

### Готовый классификатор намерений для общения клиента с чат-ботом мобильного оператора
Пользователь может использовать готовую модель для классификации намерений. Для этого используется вызов метода predict(), в котором на вход подается выражение X строкового типа.
```
predict(X)
```
### Выбор наилучшего алгоритма
Пользователь может сравнить результаты работы всех предложенных алгоритмов на своём датасете и выбрать наилучший.

### Дообучение
Пользователь может адаптировать каждый из реализованных алгоритмов для своих задач.
```
fit(X, Y, **kwargs)
```
Для адаптации готового алгоритма на вход нужно подать размеченный датасет, где X - набор выражений, Y - соответствующие им классы.

## Установка библиотеки
Прежде чем использовать библиотеку IntentClassifier убедитесь, что у вас установлен Python 3.x (мы использовали версию Python 3.6).
Выполните команду для установки всех необходимых пакетов (все необходимые пакеты перечислены в файле requirements.txt):
```
pip install -r requirements.txt
```
Кроме того, требуется скачать предобученную модель spacy xx_ent_wiki_sm с сайта <https://spacy.io/models/xx>

Для установки пакета IntentClassifier выполните команду
```
git clone https://github.com/CompTech-IntentClassifier/IntentClassifier.git
cd IntentClassifier
sudo python setup.py
```

Чтобы ознакомиться с возможностями библиотеки, запустите тесты, выполнив следующую команду:
```
python setup.py test
```

## Примеры использования

1. Использование классификатора на основе многоязычного Bert.
```
from intent_classifier import IntentClassifier
model = IntentClassifier()
sentence = "Жду ответа оператора"
cls = model.predict(sentence)
```
2. Чтобы дообучить классификатор на своих данных и использовать новую модель используйте метод fit() куда подается размеченный датасет в виде словаря состоящего из пар <класс | выражение>
```
from intent_classifier import IntentClassifier 

dataset = read_csv("newdataset.csv")
sentences = dataset['texts']
classes = dataset['classes']

# Инициализация модели алгоритма Логистическая регрессия
model = IntentClassifier('log_reg')
model = model.fit(sentences, classes)
sentence = "Жду ответа оператора"
cls = model.predict(sentence)
```
