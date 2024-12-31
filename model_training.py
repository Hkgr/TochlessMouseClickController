#model_training.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, BatchNormalization, \
    GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import RMSprop

# 1. تحميل البيانات
data = pd.read_excel("balanced_gesture_features.xlsx")

# 2. فصل الميزات (X) والتسميات (y)
X = data.iloc[:, :-1].values  # جميع الأعمدة باستثناء العمود الأخير
y = data['Gesture'].values  # العمود الأخير فقط

# تحويل الميزات إلى شكل يناسب شبكات CNN
X = X.reshape((X.shape[0], X.shape[1], 1))

# 3. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. إنشاء النموذج باستخدام CNN مع زيادة التعقيد
model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),

    Conv1D(128, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),

    Conv1D(256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),

    Conv1D(512, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),

    # إضافة طبقة GlobalAveragePooling1D لتحسين القدرة على التعميم
    GlobalAveragePooling1D(),

    # إضافة طبقة Dense أكبر
    Dense(512, activation='relu'),
    Dropout(0.6),

    Dense(256, activation='relu'),
    Dropout(0.5),

    # طبقة الإخراج
    Dense(len(np.unique(y)), activation='softmax')  # عدد الفئات
])

# 5. تجميع النموذج
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. استخدام EarlyStopping لتحسين التدريب
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# حساب أوزان الفئات لتعامل مع التوازن في البيانات
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# 7. تدريب النموذج
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping],
          class_weight=class_weights)

# 8. تقييم النموذج
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# 9. حفظ النموذج
model.save("enhanced_gesture_model.h5")
print("Enhanced model saved as enhanced_gesture_model.h5")
