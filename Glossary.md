# Основные понятия машинного обучения

1. Кросс-валидация (англ. Cross validation) - метод оценки аналитической модели и её поведения на независимых данных. При оценке модели имеющиеся в наличии данные разбиваются на k частей. Затем на k−1 частях данных производится обучение модели, а оставшаяся часть данных используется для тестирования. Процедура повторяется k раз; в итоге каждая из k частей данных используется для тестирования. В результате получается оценка эффективности выбранной модели с наиболее равномерным использованием имеющихся данных.
2. Намерение - основная цель клиента при обращении к технической поддержке.
3. Машинное обучение - класс методов искусственного интеллекта, характерной чертой которых является не прямое решение задачи, а обучение в процессе применения решений множества сходных задач.
4. Глубокое обучение - (англ. Deep learning) — совокупность методов машинного обучения (с учителем, с частичным привлечением учителя, без учителя, с подкреплением), основанных на обучении представлениям (англ. feature/representation learning), а не специализированным алгоритмам под конкретные задачи.
5. Векторизация текстовых данных - переход к векторному представлению - обработка естественного языка, направленных на сопоставление словам (и, возможно, фразам) из некоторого словаря векторов из Rn для n, значительно меньшего количества слов в словаре. Теоретической базой для векторных представлений является дистрибутивная семантика.
6. Модель - результат обучения алгоритма машинного обучения.
7. Обучение - процесс подбора весовых коэффициентов модели по размеченному датасету из входных данных и ожидаемого результата.
8. Переобучение - стадия обучения, после которой модель лишается свойства обобщения и плохо классифицирует незнакомые данные
9. DropOut - метод регуляризации. Используется рандомное принудительное обнуление весовых коэффициентов при обучении модели.