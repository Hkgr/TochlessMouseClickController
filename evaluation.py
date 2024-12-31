#evaluation.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# 1. تحميل البيانات
data = pd.read_excel("balanced_gesture_features.xlsx")

# 2. تحديد الميزات والفئات المستهدفة
X = data.drop(columns=['Gesture']).values
y = data['Gesture'].values

# 3. تطبيع البيانات
scaler = StandardScaler()
X = scaler.fit_transform(X)  # تطبيع البيانات

# 4. تحويل الفئات إلى One-Hot Encoding
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

# 5. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. إعادة تشكيل البيانات لاستخدام CNN
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))  # إعادة تشكيل بيانات الاختبار

# 7. تحميل النموذج المدرب
model = load_model("enhanced_gesture_model.h5")

# 8. توقع الفئات
y_pred = model.predict(X_test)

# 9. تحويل التوقعات والفئات الحقيقية إلى أرقام
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# 10. تقييم النموذج
accuracy = accuracy_score(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')

# 11. طباعة النتائج
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_classes, y_pred_classes))
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes))
