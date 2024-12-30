from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

model = load_model("optimized_gesture_model.h5")

X_test = np.random.rand(20, 210, 1)
y_test = np.random.randint(2, size=20)
print(f"y_test shape: {y_test.shape}")

y_pred = (model.predict(X_test, batch_size=20) > 0.5).astype("int32").flatten()  # Batch size set to 20

print(f"y_pred shape: {y_pred.shape}")

if y_test.shape[0] == y_pred.shape[0]:
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
else:
    print(f"Shape mismatch: y_test has {y_test.shape[0]} samples, but y_pred has {y_pred.shape[0]} samples.")
