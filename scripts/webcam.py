import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from scipy.spatial.distance import cdist

# Диагностика
print(f"TensorFlow version: {tf.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# Константы
IMG_SIZE = (224, 224)
MODEL_PATH = '/Users/connors/PycharmProjects/emotion_detection/models/resnet50_finetuned.h5'
CLASS_NAMES = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'uncertain']
VALENCE_AROUSAL = {
    'neutral': (0.0, 0.0), 'anger': (-0.7, 0.7), 'contempt': (-0.5, 0.3), 'disgust': (-0.6, 0.4), 'fear': (-0.4, 0.8),
    'happy': (0.7, 0.7), 'sad': (-0.7, -0.5), 'surprise': (0.3, 0.9), 'uncertain': (0.0, -0.2)
}

class EmotionDetector:
    def __init__(self, model_path, class_names=CLASS_NAMES):
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
        emotion, bbox = self.predict(frame)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

def run_webcam():
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