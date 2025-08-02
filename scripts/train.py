import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
"""
Emotion Detection Project
Цель: Разработать модель для классификации 9 эмоций и предсказания valence-arousal.

Эксперименты:
1. **Базовая модель с оверсэмплингом**:
   - RandomOverSampler для балансировки классов (50,184 сэмплов).
   - Результат: val_accuracy=0.3696 (fine-tuning), val_loss=2.2694, F1-score=~0.36.
   - Вывод: Переобучение (тренировочная точность 0.8232 >> валидационной). Оверсэмплинг повторяет данные, снижая обобщение.

2. **Без оверсэмплинга с class_weights**:
   - Использованы class_weights, оригинальный датасет (40,036 сэмплов).
   - Результат: val_accuracy=0.5115, F1-score=0.5097 (fine-tuning).
   - Вывод: Удаление оверсэмплинга и class_weights значительно улучшили обобщение.

3. **Регуляризация**:
   - Dropout 0.5 → 0.7, добавлена L2-регуляризация (0.01), Dense слой уменьшен до 128.
   - Результат: val_loss стабилизировался, val_accuracy выросла до 0.5115.
   - Вывод: Усиленная регуляризация эффективно борется с переобучением.

4. **Аугментации**:
   - Добавлены random_hue, random_saturation к flip, brightness, contrast.
   - Результат: val_accuracy=0.5115.
   - Вывод: Аугментации улучшили работу модели.

5. **Valence-Arousal модель**:
   - ResNet50 с Huber loss, линейная активация, learning rate 1e-5.
   - Результат: val_mae=0.3053, mapped_accuracy=0.3495 (было 0.3818 и 0.2649).
   - Вывод: Уменьшение learning rate и регуляризация улучшили MAE, но маппинг через cdist ограничивает точность.

6. **Fine-tuning**:
   - Разморожены все слои ResNet50, learning rate 1e-5, 15 эпох.
   - Результат: val_accuracy выросла с 0.3148 до 0.5115.
   - Вывод: Fine-tuning критически важен для повышения точности.

7. **Kaggle**:
   - Public leaderboard: 0.508 (классификационная модель).
   - Ожидание: Private leaderboard >0.4 (val_accuracy=0.5115 подтверждает).
"""
# Константы
IMG_SIZE = (224, 224)  # Размер входного изображения (для ResNet50)
BATCH_SIZE = 16  # Размер батча для оптимизации на GPU (Tesla V100)
EPOCHS = 30  # Количество эпох для начального обучения
FINE_TUNE_EPOCHS = 15  # Количество эпох для fine-tuning
CLASS_NAMES = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'uncertain']
NUM_CLASSES = len(CLASS_NAMES)  # 9 классов эмоций
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}  # Словарь для маппинга эмоций в индексы
DATA_DIR = 'train'  # Путь к тренировочным изображениям
CSV_PATH = 'train.csv'  # Путь к CSV с метками
MODEL_DIR = 'models'  # Папка для сохранения моделей
os.makedirs(MODEL_DIR, exist_ok=True)  # Создание папки models, если не существует


def augment_image(image):
    """
        Применяет аугментации к изображению для повышения робастности модели.
        Args:
            image: Входное изображение (тензор).
        Returns:
            Аугментированное изображение.
        Эксперимент: Добавлены hue, saturation для устойчивости к вариациям освещения.
        """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


def prepare_data(csv_path, img_dir):
    """
        Загружает и фильтрует данные из CSV, проверяет существование изображений.
        Args:
            csv_path: Путь к train.csv.
            img_dir: Папка с тренировочными изображениями.
        Returns:
            DataFrame с путями к изображениям и метками.
        """
    try:
        df = pd.read_csv(csv_path)
        print(f"Total images in CSV: {len(df)}")
        df['image_path'] = df['image_path'].apply(lambda x: os.path.join(img_dir, x.replace('train/', '')))
        df = df[df['image_path'].apply(os.path.exists)]  # Удаление несуществующих путей
        print(f"Valid images after filtering: {len(df)}")
        df = df[df['emotion'].isin(CLASS_NAMES)]  # Фильтрация по допустимым эмоциям
        print(f"Valid images after emotion filtering: {len(df)}")
        print(f"Class distribution:\n{df['emotion'].value_counts()}")
        return df
    except Exception as e:
        print(f"Error preparing data: {e}")
        raise


def build_model(use_valence_arousal=False):
    """
        Создает ResNet50 модель с кастомным верхним слоем.
        Args:
            use_valence_arousal: Если True, модель предсказывает (valence, arousal) с линейной активацией.
                         Если False, предсказывает 9 эмоций с softmax.
        Returns:
            Model: Скомпилированная модель.
        Эксперимент: Уменьшен Dense слой (128), добавлены Dropout 0.7, L2 (0.01).
                     Результат: val_accuracy=0.5115, снижение переобучения.
        """
    try:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.7)(x)
        outputs = Dense(2 if use_valence_arousal else NUM_CLASSES,
                        activation=None if use_valence_arousal else 'softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        for layer in base_model.layers[:20]:
            layer.trainable = False
        return model
    except Exception as e:
        print(f"Error building model: {e}")
        raise


def load_and_preprocess_image(image_path, label, class_mode='categorical'):
    """
        Загружает и предобрабатывает изображение.
        Args:
            image_path: Путь к изображению.
            label: Метка (эмоция или valence-arousal).
            class_mode: 'categorical' для классификации, 'raw' для valence-arousal.
        Returns:
            Кортеж (изображение, метка).
        """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    img = augment_image(img)

    if class_mode == 'categorical':
        label_idx = tf.reduce_sum(tf.cast(tf.equal(CLASS_NAMES, label), tf.int32) * tf.range(NUM_CLASSES))
        label = tf.one_hot(label_idx, NUM_CLASSES)
    else:
        label = tf.cast(label, tf.float32)

    return img, label


def create_dataset(df, class_mode='categorical', shuffle=True):
    """
        Создает tf.data.Dataset для обучения/валидации.
        Args:
            df: DataFrame с путями и метками.
            class_mode: 'categorical' для классификации, 'raw' для valence-arousal.
            shuffle: Перемешивать данные.
        Returns:
            tf.data.Dataset.
        Эксперимент: Добавлено tf.ensure_shape для устранения предупреждений Placeholder/_*.
        """
    if class_mode == 'raw':
        labels = np.array(df['valence_arousal'].tolist(), dtype=np.float32)
    else:
        labels = df['emotion'].values
    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'].values, labels))
    dataset = dataset.map(lambda x, y: load_and_preprocess_image(x, y, class_mode), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda x, y: (x, tf.ensure_shape(y, [None, NUM_CLASSES] if class_mode == 'categorical' else [None, 2])))
    try:
        dataset = dataset.cache()
    except:
        print("Warning: Caching disabled due to memory constraints")
    return dataset


def evaluate_model(model, dataset, class_names):
    """
        Оценивает модель, выводит F1-score и матрицу ошибок.
        Args:
            model: Обученная модель.
            dataset: Валидационный датасет.
            class_names: Список имен классов.
        Эксперимент: Добавлен F1-score для оценки качества (0.5097).
        """
    y_pred = model.predict(dataset, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in dataset])
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"F1-score: {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    """
        Основная функция: обучение классификационной и valence-arousal моделей.
        """
    try:
        print("Preparing data...")
        df = prepare_data(CSV_PATH, DATA_DIR)
        print("Sample of train.csv:\n", pd.read_csv(CSV_PATH).head())
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # Без оверсэмплинга
        print(f"Train size: {len(train_df)}")
        print(f"Class distribution:\n{train_df['emotion'].value_counts()}")

        # Valence-Arousal
        VALENCE_AROUSAL = {
            'neutral': (0.0, 0.0), 'anger': (-0.7, 0.7), 'contempt': (-0.5, 0.3), 'disgust': (-0.6, 0.4),
            'fear': (-0.4, 0.8),
            'happy': (0.7, 0.7), 'sad': (-0.7, -0.5), 'surprise': (0.3, 0.9), 'uncertain': (0.0, -0.2)
        }
        train_df['valence_arousal'] = train_df['emotion'].map(VALENCE_AROUSAL)
        val_df['valence_arousal'] = val_df['emotion'].map(VALENCE_AROUSAL)
        print("Valence-arousal sample:\n", train_df['valence_arousal'].head())

        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(train_df['emotion']), y=train_df['emotion'])
        class_weights = {i: float(w) for i, w in enumerate(class_weights)}
        print("Class weights:", class_weights)

        # Datasets
        train_dataset = create_dataset(train_df, class_mode='categorical', shuffle=True)
        val_dataset = create_dataset(val_df, class_mode='categorical', shuffle=False)
        train_va_dataset = create_dataset(train_df, class_mode='raw', shuffle=True)
        val_va_dataset = create_dataset(val_df, class_mode='raw', shuffle=False)

        print("Training ResNet50 (classification)...")
        model = build_model(use_valence_arousal=False)
        optimizer = tf.keras.optimizers.Adam(1e-4)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=2),
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
        evaluate_model(model, val_dataset, CLASS_NAMES)
        model.save(os.path.join(MODEL_DIR, 'resnet50.h5'))

        print("Fine-tuning ResNet50...")
        for layer in model.layers:
            layer.trainable = True
        optimizer = tf.keras.optimizers.Adam(1e-5)
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
        evaluate_model(model, val_dataset, CLASS_NAMES)
        model.save(os.path.join(MODEL_DIR, 'resnet50_finetuned.h5'))

        print("Training ResNet50 (valence-arousal)...")
        model_va = build_model(use_valence_arousal=True)
        optimizer = tf.keras.optimizers.Adam(1e-5)  # Уменьшен для лучшей сходимости
        model_va.compile(optimizer=optimizer,
                         loss=tf.keras.losses.Huber(),
                         metrics=['mae'])
        model_va.fit(train_va_dataset,
                     validation_data=val_va_dataset,
                     epochs=EPOCHS,
                     callbacks=callbacks)
        val_loss, val_mae = model_va.evaluate(val_va_dataset)
        print(f"Validation MAE (valence-arousal): {val_mae:.4f}")

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

        # Альтернативный маппинг через RandomForest
        from sklearn.ensemble import RandomForestClassifier
        va_class_true_idx = np.array([CLASS_TO_INDEX[e] for e in val_df['emotion'].values])
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(va_preds, va_class_true_idx)
        va_class_preds_rf = clf.predict(va_preds)
        va_acc_rf = accuracy_score(va_class_true_idx, va_class_preds_rf)
        print(f"Random Forest mapped accuracy: {va_acc_rf:.4f}")

    except Exception as e:
        print(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
