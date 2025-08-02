# Проект распознавания эмоций

Проект классифицирует 9 эмоций на изображениях лиц и включает прототип с веб-камерой. Цель: метрика > 0.29880 на Kaggle (public leaderboard) для "Зачёта" и > 0.4 (private) для высокой оценки.

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
- ~2 ГБ места для датасетов

## Установка
1. **Создайте виртуальное окружение**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
```
Обновите pip:
```bash
pip install --upgrade pip
```
Установите зависимости:
```bash
pip install -r requirements.txt
```
**Если нет GPU**:
```bash
pip install tensorflow-cpu==2.12.0
```

Скачайте датасеты:train.zip → data/train/
test.zip → data/test/
train.csv, sample_submission.csv с Kaggle → data/

Настройте PyCharm:File → New Project → Путь emotion_detection.
Укажите интерпретатор: File → Settings → Project → Python Interpreter → Add Interpreter → venv\Scripts\python.exe (Windows) или venv/bin/python (Linux/macOS).
Создайте папки data, models, scripts.

ЗапускОбучение:
```bash
python scripts/train.py
```
Обучает ResNet50 с аугментацией и fine-tuning.
Сохраняет модели в models/.

Предсказания для Kaggle:
```bash
python scripts/predict.py
```
Создаёт submission.csv в корне проекта.

Прототип с веб-камерой:
```bash
python scripts/webcam.py
```
Требуется веб-камера и models/resnet50_finetuned.h5.
Нажмите 'q' для выхода.

