import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# 1. تحميل البيانات
data = pd.read_excel("balanced_gesture_features.xlsx")

# 2. فصل الميزات (X) والتسميات (y)
X = data.iloc[:, :-1].values  # جميع الأعمدة باستثناء العمود الأخير
y = data['Gesture'].values    # العمود الأخير فقط

# تحويل الميزات إلى شكل يناسب شبكات CNN
X = X.reshape((X.shape[0], X.shape[1], 1))

# 3. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. إنشاء النموذج باستخدام CNN
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])

# 5. تجميع النموذج
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. استخدام EarlyStopping لتحسين التدريب
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 7. تدريب النموذج
model.fit(X_train, y_train, epochs=150, batch_size=1, validation_split=0.2, callbacks=[early_stopping])

# 8. تقييم النموذج
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# 9. حفظ النموذج
model.save("optimized_gesture_model.h5")
print("Optimized model saved as optimized_gesture_model.h5")
