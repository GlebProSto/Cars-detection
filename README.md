# Определитель марок и моделей автомобилей

Проект компьютерного зрения для классификации автомобилей по фотографиям на основе PyTorch и архитектуры ConvNeXt-Tiny.

Модель определяет 196 классов датасета Stanford Cars: марку, модель и год выпуска автомобиля.

## Функционал

- Загрузка датасета Stanford Cars через `kagglehub`.
- Чтение аннотаций из `.mat`-файлов через SciPy.
- Подготовка изображений в формате RGB размером `224 x 224`.
- Нормализация изображений по средним значениям и стандартным отклонениям ImageNet.
- Создание собственного класса `CustomImageDataset`.
- Transfer Learning на базе предобученной ConvNeXt-Tiny.
- Замена выходного слоя на классификатор для 196 классов.
- Обучение на GPU с AdamW, CrossEntropyLoss и смешанной точностью AMP.
- Сохранение весов и названий классов в файле `convnext_cars_196.pth`.
- Локальная проверка модели на фотографиях из папки `cars for testing`.

## Технологии

- Python
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- SciPy
- Matplotlib
- KaggleHub

## Результаты

В ноутбуке для обучения указаны следующие результаты:

- Точность на обучающей выборке: около `99.9%`.
- Точность на тестовой выборке: `86.56%`.
- Количество эпох: `15`.
- Количество классов: `196`.

Локальный скрипт [testing.py](testing.py) проверил четыре тестовых изображения:

| Фото                             | Результат |
| -------------------------------- | --------- |
| Dodge Charger Sedan 2012         | Верно     |
| Hyundai Sonata Hybrid Sedan 2012 | Верно     |
| Land Rover Range Rover SUV 2012  | Верно     |
| Plymouth Neon Coupe 1999         | Верно     |

Итог: **4 из 4 фотографий распознаны верно**.

## Структура проекта

```text
Cars detection/
├── cars for testing/
│   ├── Dodge Charger Sedan 2012.jpg
│   ├── Hyundai Sonata Hybrid Sedan 2012.jpg
│   ├── Land Rover Range Rover SUV 2012.webp
│   └── Plymouth Neon Coupe 1999.jpg
├── convnext_cars_196.pth
├── pre_trained_cars.ipynb
├── testing.py
└── README.md
```

## Описание обучения

В ноутбуке `pre_trained_cars.ipynb` выполняется загрузка датасета Stanford Cars через KaggleHub.

Из файла `cars_meta.mat` считываются названия 196 классов, а из `cars_train_annos.mat` — аннотации и названия файлов тренировочных изображений.

Данные разделены случайным образом:

- Обучающая выборка: `80%`.
- Тестовая выборка: `20%`.
- Batch size: `32`.
- Learning rate: `1e-4`.
- Optimizer: `AdamW`.
- Loss function: `CrossEntropyLoss`.
- Количество эпох: `15`.

Для обучения используется предобученная ConvNeXt-Tiny с заменённым выходным слоем:

```python
model = models.convnext_tiny(
    weights=models.ConvNeXt_Tiny_Weights.DEFAULT
)

model.classifier[2] = nn.Linear(
    model.classifier[2].in_features,
    196
)
```

Для ускорения обучения на GPU используется PyTorch AMP.

## Подготовка окружения

Рекомендуется Python 3.10 или новее.

Установите необходимые библиотеки:

```bash
pip install torch torchvision pillow
```

Для запуска ноутбука дополнительно потребуются:

```bash
pip install kagglehub scipy scikit-learn pandas numpy matplotlib
```

## Локальное тестирование

Файл весов `convnext_cars_196.pth` должен находиться в корне проекта.

Тестовые фотографии помещаются в папку `cars for testing`.

Название тестовой фотографии используется как ожидаемый класс:

```text
Dodge Charger Sedan 2012.jpg
```

Запустите проверку:

```bash
cd "D:\deep learning\Cars detection"
python testing.py
```

Скрипт выполняет следующее:

1. Загружает чекпоинт модели.
2. Восстанавливает ConvNeXt-Tiny с 196 выходными классами.
3. Преобразует фотографии в формат `224 x 224`.
4. Выполняет предсказание и применяет `softmax`.
5. Выводит ожидаемый класс, предсказанный класс и уверенность.
6. Считает количество правильно распознанных фотографий.

## Возможные улучшения

- Добавить аугментацию данных: повороты, отражения и изменение яркости.
- Использовать планировщик learning rate.
- Сделать стратифицированное разделение данных.
- Добавить проверку на разных типах изображений и фоне.
- Создать веб-интерфейс на Streamlit или FastAPI.
- Добавить возможность загрузки фотографии через интерфейс.

🍿 Демонстрация
<img width="418" height="553" alt="Снимок экрана 2026-11-22 110936" src="https://github.com/user-attachments/assets/beaeaf0a-d32a-4980-8bc4-30720696c406" />

