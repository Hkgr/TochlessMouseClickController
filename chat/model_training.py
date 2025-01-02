# model_builder.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class GestureModelBuilder:
    """
    بناء نموذج CNN محسن للتعرف على الإيماءات باستخدام البيانات المتوازنة
    المميزات:
    - يتعامل مع 424 ميزة مدخلة (Feature_1 to Feature_424)
    - يدعم البيانات المتوازنة باستخدام SMOTE
    - معمارية CNN محسنة للتصنيف
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def prepare_data(self, data_file):
        """تحضير البيانات المتوازنة"""
        # تحميل البيانات المتوازنة مسبقاً
        data = pd.read_excel(data_file)
        data['Gesture'] = data['Gesture'].map(lambda x: x - 1)
        print(f"شكل البيانات: {data.shape}")

        # فصل الميزات والتصنيفات
        X = data.drop('Gesture', axis=1).values
        y = data['Gesture'].values

        # تطبيع البيانات
        X = self.scaler.fit_transform(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # إعادة التشكيل لـ CNN

        print(f"شكل المدخلات النهائي: {X.shape}")
        print(f"عدد الفئات الفريدة: {len(np.unique(y))}")

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def build_model(self, input_shape, num_classes):
        model = Sequential([
            # تحسين Block 1
            Conv1D(64, 5, padding='same', activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            # تحسين Block 2
            Conv1D(128, 3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.4),

            # تحسين Block 3
            Conv1D(256, 3, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.4),

            # Classification Layers
            GlobalAveragePooling1D(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )


        # تحسين المعايير
        optimizer = Adam(
            learning_rate=0.0005,  # learning rate أقل للتدريب المستقر
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True  # تفعيل AMSGrad للتحسين
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def get_callbacks(self):
        """إعداد callbacks للتدريب"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

    def save_model(self, filepath):
        """حفظ النموذج"""
        if self.model:
            self.model.save(filepath)
            print(f"تم حفظ النموذج في {filepath}")
        else:
            print("يجب بناء النموذج أولاً")


# مثال على الاستخدام
if __name__ == "__main__":
    builder = GestureModelBuilder()

    # تحضير البيانات
    X_train, X_test, y_train, y_test = builder.prepare_data("balanced_gesture_features.xlsx")

    # بناء النموذج
    model = builder.build_model(
        input_shape=(X_train.shape[1], 1),
        num_classes=len(np.unique(y_train))
    )

    # طباعة ملخص النموذج
    print(model.summary())

    # إعداد callbacks
    callbacks = builder.get_callbacks()

    # تدريب النموذج
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,  # زيادة حجم الدفعة
     #   class_weight='auto',  # إضافة أوزان للفئات
        callbacks=callbacks,
        verbose=1
    )

    # حفظ النموذج
    builder.save_model("gesture_model.h5")