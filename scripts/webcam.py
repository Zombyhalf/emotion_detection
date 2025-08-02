import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from scipy.spatial.distance import cdist
"""
Emotion Detection Webcam Prototype
Цель: Детекция эмоций в реальном времени через веб-камеру с использованием ResNet50.
Метрики:
- Классификационная модель: val_accuracy=0.5115, стабильные предсказания.
- Valence-Arousal: mapped_accuracy=0.3495, частые 'neutral' из-за маппинга.
Эксперименты:
1. Valence-Arousal модель:
   - Результат: Частые предсказания 'neutral' (mapped_accuracy=0.3495).
   - Вывод: Маппинг через cdist ограничивает качество.
2. Классификационная модель:
   - Результат: Более точные и стабильные предсказания (val_accuracy=0.5115).
   - Вывод: Предпочтительна для прототипа.
3. Предобработка:
   - Замена деления на 255.0 на preprocess_input.
   - Результат: Улучшение точности детекции.
"""

# Константы
IMG_SIZE = (224, 224)  # Размер входного изображения (для ResNet50)
MODEL_PATH = 'models/resnet50_finetuned.h5'  # Путь к модели, можно менять на resnet50_valence_arousal.h5
CLASS_NAMES = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'uncertain']
VALENCE_AROUSAL = {
    'neutral': (0.0, 0.0), 'anger': (-0.7, 0.7), 'contempt': (-0.5, 0.3), 'disgust': (-0.6, 0.4), 'fear': (-0.4, 0.8),
    'happy': (0.7, 0.7), 'sad': (-0.7, -0.5), 'surprise': (0.3, 0.9), 'uncertain': (0.0, -0.2)
}

class EmotionDetector:
    """
    Класс для детекции эмоций на видео с веб-камеры.
    """
    def __init__(self, model_path, class_names=CLASS_NAMES):
        """
        Инициализация модели и детектора лиц.
        Args:
            model_path: Путь к модели (классификационная или valence-arousal).
            class_names: Список эмоций.
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.class_names = class_names
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.is_valence_arousal = 'valence_arousal' in model_path
            print(f"Model type: {'Valence-Arousal' if self.is_valence_arousal else 'Classification'}")
            if self.face_cascade.empty():
                raise ValueError("Failed to load Haar cascade")
        except Exception as e:
            print(f"Error initializing EmotionDetector: {e}")
            raise

    def preprocess_frame(self, frame):
        """
               Предобработка кадра: детекция лица, изменение размера, нормализация.
               Args:
                   frame: Входной кадр (BGR).
               Returns:
                   Кортеж (предобработанное изображение, координаты лица).
               Эксперимент: preprocess_input заменил деление на 255.0 для согласованности с train.py.
               """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                return None, None
            (x, y, w, h) = faces[0]
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, IMG_SIZE)
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)  # Согласован с train.py
            return face, (x, y, w, h)
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None, None

    def predict(self, frame):
        """
                Предсказание эмоции на кадре.
                Args:
                    frame: Входной кадр (BGR).
                Returns:
                    Кортеж (эмоция, координаты лица).
                """
        processed_frame, bbox = self.preprocess_frame(frame)
        if processed_frame is None:
            return "No face detected", None
        predictions = self.model.predict(processed_frame, verbose=0)
        if self.is_valence_arousal:
            # Valence-Arousal: маппинг через cdist
            if predictions.shape != (1, 2):
                print(f"Warning: Unexpected prediction shape: {predictions.shape}")
                return "neutral", bbox
            va_values = np.array(list(VALENCE_AROUSAL.values()))
            emotions = list(VALENCE_AROUSAL.keys())
            emotion = emotions[np.argmin(cdist(predictions, va_values))]
        else:
            # Классификация
            if predictions.shape != (1, len(self.class_names)):
                print(f"Warning: Unexpected prediction shape: {predictions.shape}")
                return "neutral", bbox
            emotion = self.class_names[np.argmax(predictions)]
        return emotion, bbox

    def display_emotion(self, frame):
        """
                Отображение эмоции и рамки вокруг лица на кадре.
                Args:
                    frame: Входной кадр (BGR).
                Returns:
                    Кадр с рамкой и текстом эмоции.
                """
        emotion, bbox = self.predict(frame)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

def run_webcam():
    """
        Запуск веб-камеры для детекции эмоций в реальном времени.
        Нажмите 'q' для выхода.
        """
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Failed to open webcam")
        detector = EmotionDetector(model_path=MODEL_PATH)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            frame = detector.display_emotion(frame)
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in run_webcam: {e}")
        raise

if __name__ == "__main__":
    run_webcam()
