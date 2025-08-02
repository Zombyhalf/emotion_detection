import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from scipy.spatial.distance import cdist

# Диагностика
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# Константы
IMG_SIZE = (224, 224)
TEST_DIR = '/Users/connors/PycharmProjects/emotion_detection/data/test'
MODEL_PATH = '/Users/connors/PycharmProjects/emotion_detection/models/resnet50_valence_arousal.h5'
CLASS_NAMES = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'uncertain']
VALENCE_AROUSAL = {
    'neutral': (0.0, 0.0), 'anger': (-0.7, 0.7), 'contempt': (-0.5, 0.3), 'disgust': (-0.6, 0.4), 'fear': (-0.4, 0.8),
    'happy': (0.7, 0.7), 'sad': (-0.7, -0.5), 'surprise': (0.3, 0.9), 'uncertain': (0.0, -0.2)
}
USE_TEST_PREFIX = False  # Установите True, если Kaggle ожидает image_path вида 'test/0.jpg'

def generate_submission():
    try:
        # Загрузка модели
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)

        # Определение типа модели
        is_valence_arousal = 'valence_arousal' in MODEL_PATH
        print(f"Model type: {'Valence-Arousal' if is_valence_arousal else 'Classification'}")

        # Подготовка тестовых путей
        test_image_paths = [f"test/{i}.jpg" if USE_TEST_PREFIX else f"{i}.jpg" for i in range(5000)]
        predictions = []

        for img_path in test_image_paths:
            full_path = os.path.join(TEST_DIR, img_path.lstrip('test/'))  # Убираем 'test/' для локального пути
            if not os.path.exists(full_path):
                print(f"Warning: Image not found: {full_path}")
                predictions.append('neutral')
                continue

            # Загрузка и предобработка изображения
            img = load_img(full_path, target_size=IMG_SIZE)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)  # Согласован с train.py

            # Предсказание
            pred = model.predict(img, verbose=0)

            if is_valence_arousal:
                # Valence-Arousal модель: маппинг через cdist
                if pred.shape != (1, 2):
                    print(f"Warning: Unexpected prediction shape for {img_path}: {pred.shape}")
                    predictions.append('neutral')
                else:
                    va_values = np.array(list(VALENCE_AROUSAL.values()))
                    emotions = list(VALENCE_AROUSAL.keys())
                    emotion = emotions[np.argmin(cdist(pred, va_values))]
                    predictions.append(emotion)
            else:
                # Классификационная модель
                if pred.shape != (1, len(CLASS_NAMES)):
                    print(f"Warning: Unexpected prediction shape for {img_path}: {pred.shape}")
                    predictions.append('neutral')
                else:
                    emotion = CLASS_NAMES[np.argmax(pred)]
                    predictions.append(emotion)

        # Создание submission.csv
        submission = pd.DataFrame({
            'image_path': test_image_paths,
            'emotion': predictions
        })
        submission_path = 'submission.csv'
        submission.to_csv(submission_path, index=False)
        print(f"Submission file created: {submission_path}")
        print("First 5 rows of submission.csv:\n", pd.read_csv(submission_path).head())

    except Exception as e:
        print(f"Error in generate_submission: {e}")
        raise

if __name__ == "__main__":
    generate_submission()

#
#
#
# import os
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
#
# # Диагностика
# print(f"TensorFlow version: {tf.__version__}")
# print(f"NumPy version: {np.__version__}")
#
# # Константы
# IMG_SIZE = (224, 224)  # Согласовать с train.py
# TEST_DIR = '/Users/connors/PycharmProjects/emotion_detection/data/test'
# MODEL_PATH = '/Users/connors/PycharmProjects/emotion_detection/models/resnet50_finetuned.h5'
# CLASS_NAMES = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'uncertain']
#
#
# def generate_submission():
#     try:
#         model = tf.keras.models.load_model(MODEL_PATH)
#         test_image_paths = [f"{i}.jpg" for i in range(5000)]
#         predictions = []
#
#         for img_path in test_image_paths:
#             full_path = os.path.join(TEST_DIR, img_path)
#             if not os.path.exists(full_path):
#                 print(f"Warning: Image not found: {full_path}")
#                 predictions.append('neutral')
#                 continue
#             img = load_img(full_path, target_size=IMG_SIZE)
#             img = img_to_array(img) / 255.0
#             img = np.expand_dims(img, axis=0)
#             pred = model.predict(img, verbose=0)
#             emotion = CLASS_NAMES[np.argmax(pred)]
#             predictions.append(emotion)
#
#         submission = pd.DataFrame({'image_path': test_image_paths, 'emotion': predictions})
#         submission.to_csv('/Users/connors/PycharmProjects/emotion_detection/submission.csv', index=False)
#         print("Submission file created: submission.csv")
#     except Exception as e:
#         print(f"Error in generate_submission: {e}")
#         raise
#
#
# if __name__ == "__main__":
#     generate_submission()