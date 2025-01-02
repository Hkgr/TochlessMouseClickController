# GestureRecognizer.py
from tensorflow.keras.models import load_model
from collections import Counter
import numpy as np


class GestureRecognizer:
    def __init__(self, model_path, classes, smoothing_window=5, confidence_threshold=0.6):
        """
        تهيئة نظام التعرف على الإيماءات
        """
        self.model = load_model(model_path)
        self.classes = classes
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.gesture_history = []

    def preprocess_data(self, hand_data):
        """
        معالجة بيانات اليد قبل إدخالها للنموذج
        """
        if len(hand_data) != 42:
            return None

        # تطبيع البيانات
        data = np.array(hand_data, dtype=np.float32)
        data = (data - np.mean(data)) / np.std(data)
        return data.reshape(1, 42, 1)

    def predict_gesture(self, data):
        # طباعة شكل البيانات قبل المعالجة للتأكد
        print("Original data shape:", data.shape)

        # التحقق من أن البيانات يمكن إعادة تشكيلها للشكل المتوقع
        try:
            processed_data = data.reshape(-1, 21, 2)  # تغيير الشكل ليتوافق مع النموذج
        except ValueError as e:
            print(f"Error reshaping data: {e}")
            return None, None

        # طباعة شكل البيانات بعد المعالجة
        print("Processed data shape:", processed_data.shape)

        # التنبؤ باستخدام النموذج
        prediction = self.model.predict(processed_data, verbose=0)

        # استخراج اسم الإشارة ونسبة الثقة
        gesture_name = np.argmax(prediction, axis=1)  # قد تحتاج لتغيير هذا حسب منطقك
        confidence = np.max(prediction, axis=1)

        return gesture_name, confidence

    def reset_history(self):
        """
        إعادة تعيين تاريخ الإيماءات
        """
        self.gesture_history.clear()