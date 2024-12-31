import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical

# تحميل البيانات من الملف الأفضل
data = pd.read_excel("balanced_gesture_features.xlsx")

# تحديد الميزات (X) والفئات المستهدفة (y)
X = data.drop(columns=['Gesture'])  # استبدل 'Gesture' باسم العمود الذي يحتوي على الفئات إذا كان مختلفًا
y = data['Gesture']

# تطبيع الميزات لضمان أن القيم في نطاق ثابت
X = X / X.max().values

# تحويل الفئات إلى تنسيق One-Hot Encoding إذا كانت متعددة التصنيفات
num_classes = len(y.unique())
y = to_categorical(y, num_classes=num_classes)

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إعادة تشكيل البيانات إذا كانت بحاجة إلى قناة إضافية (لشبكات CNN مثلاً)
X_test = np.expand_dims(X_test.values, axis=-1)  # فقط لاختبار البيانات

# تحميل النموذج المدرب
model = load_model("optimized_gesture_model.h5")

# توقع الفئات على مجموعة الاختبار
y_pred = model.predict(X_test)

# إذا كانت y_pred أحادية البُعد، يجب تحويلها إلى مصفوفة ثنائية البُعد
if len(y_pred.shape) == 1:
    y_pred_classes = y_pred  # إذا كانت بالفعل أحادية البُعد، لا نحتاج لتحويلها
else:
    y_pred_classes = np.argmax(y_pred, axis=1)  # اختيار التصنيف الأكثر احتمالية

# تحويل الفئات الحقيقية إلى تنسيق أحادي البُعد
y_test_classes = np.argmax(y_test, axis=1)

# حساب الدقة و F1 Score
accuracy = accuracy_score(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')  # استخدم 'weighted' للفئات المتعددة

# طباعة النتائج
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_classes, y_pred_classes))
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes))
