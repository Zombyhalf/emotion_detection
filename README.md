# Проект распознавания эмоций

Проект классифицирует 9 эмоций на изображениях лиц и включает прототип с веб-камерой. Цель: метрика > 0.29880 на Kaggle (public leaderboard) для "Зачёта" и > 0.4 (private) для высокой оценки. Файлы для обучения по адресу https://www.kaggle.com/competitions/skillbox-computer-vision-project/overview 

## Структура проекта

/emotion_detection
├── data/
│   ├── train/                # Тренировочные изображения (из train.zip)
│   ├── test/                 # Тестовые изображения (из test.zip)
│   ├── train.csv             # Метки для обучения
│   ├── sample_submission.csv # Шаблон для Kaggle
├── models/
│   ├── resnet50.h5           # Веса модели
│   ├── resnet50_finetuned.h5 # Дообученная модель
│   ├── resnet50_valence_arousal.h5 # Дообученная модель valence_arousal
├── scripts/
│   ├── train.py              # Обучение модели
│   ├── predict.py            # Генерация предсказаний
│   ├── webcam.py             # Прототип с веб-камерой
├── requirements.txt          # Зависимости
├── README.md                # Инструкции

## Требования
- Python 3.8–3.10
- PyCharm Community Edition
- Веб-камера (для прототипа)
- ~3 ГБ места для датасетов

## Установка

Установите зависимости:
```bash
pip install -r requirements.txt
```

## Подготовка данных
- **Датасет**: Скачайте датасет с [Kaggle/другой источник] и разместите:
  - Тренировочные изображения в `data/train/`.
  - Тестовые изображения в `data/test/`.
  - `train.csv` в `data/` (формат: `image_path`, `emotion`).
- **Модели**: Обучите модели с помощью `train.py`, чтобы сгенерировать:
  - `models/resnet50.h5`
  - `models/resnet50_finetuned.h5`
  - `models/resnet50_valence_arousal.h5`

## Предсказания для Kaggle:
```bash
python scripts/predict.py
```
Создаёт submission.csv в корне проекта.

## Прототип с веб-камерой:
```bash
python scripts/webcam.py
```
Требуется веб-камера и models/resnet50_finetuned.h5.
Нажмите 'q' для выхода.

