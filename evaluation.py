from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# تحميل النموذج
model = load_model("gesture_model.h5")

# بيانات التقييم الوهمية
X_test = np.random.rand(20, 20)  # 20 عينة
y_test = np.random.randint(2, size=20)

# التنبؤ
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# حساب المقاييس
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
