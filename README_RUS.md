## IntentClasiifier
 - библиотека предобученных алгоритмов для классификации намерений человека при общении человека с чат ботом. Библиотека была разработана в рамках проекта на зимней школе CompTech@NSK в Новосибирске, 2019. Идея проекта была софрмулирвоана
 Задачи бибилиотеки:
 * выполнять классификацию русскоязычных интентов с высокой точностью и за малое время *
 * выполнять дообучения на новом датасете 
 * предоставлять разработчику выбор между различными алгоритмами
 
 ## Структура библиотеки
 В библиотеке представленны следующие алгоритмы машинного обучения. 
* 
* 
* 
* 
* 
* 

Модель были разработаны специально для работы с небольшими обучающими датасетами небольших текстов.
Каждый из алгоритмов был предобучен на датасете состоящем из русскоязычных обращения пользователя к чатботу мобильного оператора. Датасет разделен на 10 классов и состоит примерно из 900 выражений.

Обучение и тестирование алгоритмов проводилось с использование кросс-валидации на 10 блоках. Подробные результаты тестирования представлены в документе Результаты тестов.


## Возможности для использования:
* предложенную в первой версии библиотеку можно использовать для реализации чатботов мобильных операторов для общения с пользователями.
* переобученную модель можно использовать для 
** реализации любых других чатботов, например чатботов для заказа покупок на дом, в приложении банка и т.п.;
** для классификации заголовков газет, рекламы;
** автоматической классификации сообщений в почтовом ящике по теме и части текста;
** реализации советчика при выборе фильма, книги, еды, места и т.п;
** и другое.
На основе библиотеки создан чатбот для платформы ВКонтакте, 

## Возможности библиотеки
В каждом алгоритме реализованы слледующие функции
# Дообучение:
*fit(self, X, y, **kwargs)
Для дообучения готовой нейронной сети на вход нужно подать размеченный датасет, X - набор ввыраженийб н - соответствующие им классы.

# Классификация намерений:
*predict(self, X):
Для использования готовой классификации намерений вызовите метод predict(), подав на вход выражение X, type string.

## Установка
```
git clone https://github.com/CompTech-IntentClassifier/IntentClassifier.git
cd IntentClassifier
sudo python setup.py
```

But I recommend you to use pip and install this package from PyPi:

```
pip install comp_tech_intent_classifier
```

Чтобы ознакомиться с вохможностями библиотеки есть возможность запустить тесты

```
python setup.py test
```

## Использование

Исопльзование классификатора на основе многоязычного Bert.
```
from IntentClassifier import BertClf
model = BertClf()
sentence = "Жду ответа оператора"
class = model.predict(sentence)
```
Чтобы дообучить классификатор на своих данных и использовать новую модель в дальнейшем используйте метод fit и подайте ей размеченный датасет в виде словаря класс - пример выражениея
```
from IntentClassifier import BertClf
dataset = read_csv("newdataset.csv")
sentences = dataset['texts']
classes = dataset['classes']

model = BertClf()
model.fit(sentences, classes)
sentence = "Жду ответа оператора"
class = model.predict(sentence)
```