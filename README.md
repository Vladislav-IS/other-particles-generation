# Генерация "остальных" частиц (модель Pion-GAN)

## Структура репозитория

```
other-particles-generation/
├── notebooks/                          # IPython-блокноты с экспериментами
│   ├── cond_experiment.ipynb           # Подсчет метрик качества для Pion-GAN с различных внешним условием
│   ├── flows_experiment.ipynb          # Обучение и дообучение "потоковой" версии модели
│   ├── models_experiment.ipynb         # Подсчет метрик качества для разных моделей 
│   └── n_modes_experiment.ipynb        # Подсчет метрик качества для Pion-GAN с различным количеством мод в смеси гауссовских шумов
├── src/                                # Исходники
│   ├── __init__.py
│   ├── сalculations.py                 # Функции физических расчетов и предобработки данных ROOT-файлов 
│   ├── config.py                       # Глобальные переменные с фиксированным значением (коды частиц и т.п.)
│   ├── pion_gan.py                     # Реализация модели Pion-GAN
│   ├── gaussian_mixture.py             # Реализация обучаемой смеси гауссовских шумов
│   ├── inference.py                    # Функции расчета метрик
│   ├── plots.py                        # Функции построения графиков
│   └── training.py                     # Функции обучения моделей разных типов
├── metrics/                            # Значения метрик, полученные в ходе экспериментов
│   ├── models_metrics.csv              # Метрики, полученные в models_experiment.ipynb
│   ├── n_modes_metrics.csv             # Метрики, полученные в n_modes_experiment.ipynb
│   ├── cond_metrics.csv                # Метрики, полученные в cond_experiment.ipynb
│   └── flows_ls.csv                    # Метрики, полученные в flows_experiment.ipynb
├── trained_models/                     # Веса обученных моделей
│   ├── _best_urqmd*                    # Веса "потоковой" модели
│   ├── _best_smash*                    # Веса "потоковой" модели, дообученной на SMASH
│   ├── B*                              # Веса модели, полученной в cond_experiment.ipynb, с прицельным параметром в качестве внешнего условия
│   └── pion_gan*                       # Веса базовой модели Pion-GAN, полученные в models_experiment.ipynb
├── requirements.txt                    # Зависимости для src
└── README.md                           # Описание проекта
```

## Обучающие данные

Обучающие данные находятся по общедоступным ссылкам Google Drive.

UrQMD ($b = 1.7$ фм):
```
https://drive.google.com/uc?id=1EhDJ0DSe1AuHNRxUIbQV5r8AXo4TP8Kp
```

UrQMD ($b = 5$ фм):
```
https://drive.google.com/uc?id=1mBNT9X2qJRgRFHAhRIfv6kFSHQRnbZ4T
```

SMASH:
```
https://drive.google.com/uc?id=1BrMboOkpiwmMfJO0PStSxUmiLqWqYHi_
```

Примеры загрузки данных с помощью `gdown` содержатся в ipynb-файлах (папка `notebooks`) и `example.py`.

## Запуск кода

1. Установка зависимостей: `pip install -r requirements.txt`.

2. Запуск примера: `python ./example.py`.
