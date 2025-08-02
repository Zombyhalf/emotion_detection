import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

# Диагностика
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Physical devices: {tf.config.list_physical_devices()}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Built with GPU support: {tf.test.is_built_with_gpu_support()}")

# Константы
IMG_SIZE = (160, 160)  # M2: Memory-efficient
BATCH_SIZE = 8  # M2: GPU-optimized
EPOCHS = 40
FINE_TUNE_EPOCHS = 20
CLASS_NAMES = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'uncertain']
NUM_CLASSES = len(CLASS_NAMES)
# Local paths
DATA_DIR = '/Users/connors/PycharmProjects/emotion_detection/data/train'
CSV_PATH = '/Users/connors/PycharmProjects/emotion_detection/data/train.csv'
MODEL_DIR = '/Users/connors/PycharmProjects/emotion_detection/models'
# Kaggle paths: uncomment for Kaggle
# DATA_DIR = '/kaggle/working/data/train'
# CSV_PATH = '/kaggle/working/data/train.csv'
# MODEL_DIR = '/kaggle/working/models'
os.makedirs(MODEL_DIR, exist_ok=True)

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image

def prepare_data(csv_path, img_dir):
    try:
        df = pd.read_csv(csv_path)
        df['image_path'] = df['image_path'].apply(lambda x: os.path.join(img_dir, x.replace('train/', '')))
        df = df[df['image_path'].apply(os.path.exists)]
        df = df[df['emotion'].isin(CLASS_NAMES)]
        print(f"Loaded {len(df)} valid images")
        print(f"Class distribution:\n{df['emotion'].value_counts()}")
        return df
    except Exception as e:
        print(f"Error preparing data: {e}")
        raise

def build_model(use_valence_arousal=False):
    try:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)  # Reduced for M2
        x = Dropout(0.5)(x)
        outputs = Dense(2 if use_valence_arousal else NUM_CLASSES, activation='tanh' if use_valence_arousal else 'softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        for layer in base_model.layers[:40]:
            layer.trainable = False
        return model
    except Exception as e:
        print(f"Error building model: {e}")
        raise

def load_and_preprocess_image(image_path, label, class_mode='categorical'):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    img = augment_image(img)
    if class_mode == 'categorical':
        label = tf.one_hot(tf.argmax(tf.cast(tf.equal(CLASS_NAMES, label), tf.int32)), NUM_CLASSES)
    else:  # valence-arousal
        label = tf.cast(label, tf.float32)
    return img, label

def create_dataset(df, class_mode='categorical', shuffle=True):
    if class_mode == 'raw':
        labels = np.array(df['valence_arousal'].tolist(), dtype=np.float32)
    else:
        labels = df['emotion'].values
    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'].values, labels))
    dataset = dataset.map(lambda x, y: load_and_preprocess_image(x, y, class_mode), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=500)  # Reduced for M2
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # Cache only if memory allows
    try:
        dataset = dataset.cache()
    except:
        print("Warning: Caching disabled due to memory constraints")
    return dataset

def main():
    try:
        print("Preparing data...")
        df = prepare_data(CSV_PATH, DATA_DIR)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # Oversampling
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(train_df[['image_path']], train_df['emotion'])
        train_df = pd.DataFrame({
            'image_path': X_resampled['image_path'],
            'emotion': y_resampled
        })
        print(f"Train size after oversampling: {len(train_df)}")
        print(f"Class distribution after oversampling:\n{train_df['emotion'].value_counts()}")

        # Valence-Arousal
        VALENCE_AROUSAL = {
            'neutral': (0.0, 0.0), 'anger': (-0.7, 0.7), 'contempt': (-0.5, 0.3), 'disgust': (-0.6, 0.4), 'fear': (-0.4, 0.8),
            'happy': (0.7, 0.7), 'sad': (-0.7, -0.5), 'surprise': (0.3, 0.9), 'uncertain': (0.0, -0.2)
        }
        train_df['valence_arousal'] = train_df['emotion'].map(VALENCE_AROUSAL)
        val_df['valence_arousal'] = val_df['emotion'].map(VALENCE_AROUSAL)

        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(train_df['emotion']), y=train_df['emotion'])
        class_weights = {i: float(w) for i, w in enumerate(class_weights)}

        # Datasets
        train_dataset = create_dataset(train_df, class_mode='categorical', shuffle=True)
        val_dataset = create_dataset(val_df, class_mode='categorical', shuffle=False)
        train_va_dataset = create_dataset(train_df, class_mode='raw', shuffle=True)
        val_va_dataset = create_dataset(val_df, class_mode='raw', shuffle=False)

        print("Training ResNet50 (classification)...")
        model = build_model(use_valence_arousal=False)
        optimizer = tf.keras.optimizers.legacy.Adam(1e-3)  # M2-compatible
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint(os.path.join(MODEL_DIR, 'resnet50_best_weights.h5'),
                            save_best_only=True, save_weights_only=True)
        ]
        model.fit(train_dataset,
                  validation_data=val_dataset,
                  epochs=EPOCHS,
                  class_weight=class_weights,
                  callbacks=callbacks)
        val_loss, val_acc = model.evaluate(val_dataset)
        print(f"Validation accuracy (classification): {val_acc:.4f}")
        model.save(os.path.join(MODEL_DIR, 'resnet50.h5'))

        print("Fine-tuning ResNet50...")
        for layer in model.layers[:20]:
            layer.trainable = True
        optimizer = tf.keras.optimizers.legacy.Adam(3e-4)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_dataset,
                  validation_data=val_dataset,
                  epochs=FINE_TUNE_EPOCHS,
                  class_weight=class_weights,
                  callbacks=callbacks)
        val_loss, val_acc = model.evaluate(val_dataset)
        print(f"Validation accuracy (fine-tuned): {val_acc:.4f}")
        model.save(os.path.join(MODEL_DIR, 'resnet50_finetuned.h5'))

        print("Training ResNet50 (valence-arousal)...")
        model_va = build_model(use_valence_arousal=True)
        optimizer = tf.keras.optimizers.legacy.Adam(1e-3)
        model_va.compile(optimizer=optimizer,
                         loss='mse',
                         metrics=['mae'])
        model_va.fit(train_va_dataset,
                     validation_data=val_va_dataset,
                     epochs=EPOCHS,
                     callbacks=callbacks)
        val_loss, val_mae = model_va.evaluate(val_va_dataset)
        print(f"Validation MAE (valence-arousal): {val_mae:.4f}")
        model.save(os.path.join(MODEL_DIR, 'resnet50_valence_arousal.h5'))

        # Valence-Arousal mapping
        from scipy.spatial.distance import cdist
        va_preds = model_va.predict(val_va_dataset, verbose=0)
        va_values = np.array(list(VALENCE_AROUSAL.values()))
        emotions = list(VALENCE_AROUSAL.keys())
        va_class_preds = [emotions[np.argmin(cdist([va_preds[i]], va_values))] for i in range(len(va_preds))]
        va_class_true = val_df['emotion'].values
        from sklearn.metrics import accuracy_score
        va_acc = accuracy_score(va_class_true, va_class_preds)
        print(f"Valence-Arousal mapped accuracy: {va_acc:.4f}")

    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
