# gesture_cnn_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_data(data):
    """
    """
    print("\n=== تحليل البيانات ===")
    print(f"شكل البيانات: {data.shape}")
    print("\nتوزيع الفئات:")
    print(data['Gesture'].value_counts())
    print("\nإحصائيات الميزات:")
    print(data.describe())

    missing = data.isnull().sum()
    if missing.any():
        print("\nالقيم المفقودة:")
        print(missing[missing > 0])

    print("\nفحص القيم الشاذة...")
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    print("عدد القيم الشاذة في كل عمود:")
    print(outliers[outliers > 0])

    plt.figure(figsize=(10, 5))
    data['Gesture'].value_counts().plot(kind='bar')
    plt.title('توزيع الفئات')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()

    return data['Gesture'].value_counts()


def prepare_data(file_path='balanced_gesture_features.xlsx'):
    """
    """
    print("\n=== تجهيز البيانات ===")
    data = pd.read_excel(file_path)

    class_distribution = analyze_data(data)

    variance = data.var()
    low_variance_cols = variance[variance < 0.01].index
    if len(low_variance_cols) > 0:
        print(f"\nالأعمدة ذات التباين المنخفض: {low_variance_cols}")
        data = data.drop(columns=low_variance_cols)

    X = data.drop('Gesture', axis=1).values
    y = data['Gesture'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42,
        stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nأبعاد بيانات التدريب: {X_train.shape}")
    print(f"أبعاد بيانات الاختبار: {X_test.shape}")

    return X_train, X_test, y_train, y_test, le, scaler


class GestureRecognitionCNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """
        """
        model = models.Sequential([
            layers.Input(shape=(self.input_shape,)),
            layers.Reshape((self.input_shape, 1)),

            # Block 1
            layers.Conv1D(32, 3, padding='same'),
            layers.LeakyReLU(0.1),
            layers.BatchNormalization(),
            layers.Conv1D(32, 3, padding='same'),
            layers.LeakyReLU(0.1),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            # Block 2
            layers.Conv1D(64, 3, padding='same'),
            layers.LeakyReLU(0.1),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, padding='same'),
            layers.LeakyReLU(0.1),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            # Dense layers
            layers.Flatten(),
            layers.Dense(128),
            layers.LeakyReLU(0.1),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model

    def compile_model(self, learning_rate=0.000005):
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val, class_weights=None, epochs=100, batch_size=16):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[early_stopping, reduce_lr]
        )
        return history


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    """
    print("\n=== تقييم النموذج ===")

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred == y_true)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # تقرير التصنيف التفصيلي
    print("\nتقرير التصنيف التفصيلي:")
    print(classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_
    ))

    plt.figure(figsize=(10, 8))
    confusion = tf.math.confusion_matrix(y_true, y_pred)
    sns.heatmap(
        confusion.numpy(),
        annot=True,
        fmt='d',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('مصفوفة الالتباس')
    plt.ylabel('القيمة الحقيقية')
    plt.xlabel('التنبؤ')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def main():
    X_train, X_test, y_train, y_test, label_encoder, scaler = prepare_data()

    class_weights = dict(enumerate(
        len(y_train) / (len(np.unique(y_train)) * np.sum(y_train, axis=0))
    ))

    model = GestureRecognitionCNN(
        input_shape=X_train.shape[1],
        num_classes=len(label_encoder.classes_)
    )
    model.compile_model(learning_rate=0.001)

    history = model.train(
        X_train, y_train,
        X_test, y_test,
        class_weights=class_weights,
        epochs=300,
        batch_size=16
    )


    model.model.save('gesture_cnn_model.h5')
    np.save('scaler_params.npy', {
        'mean': scaler.mean_,
        'scale': scaler.scale_
    })


if __name__ == "__main__":
    main()