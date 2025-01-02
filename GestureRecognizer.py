# GestureRecognizer.py
from tensorflow.keras.models import load_model
from collections import Counter
import numpy as np


class GestureRecognizer:
    def __init__(self, model_path, classes, smoothing_window=5, confidence_threshold=0.6):
        """
        """
        self.model = load_model(model_path)
        self.classes = classes
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.gesture_history = []

    def preprocess_data(self, hand_data):
        """
        """
        if len(hand_data) != 42:
            return None

        data = np.array(hand_data, dtype=np.float32)
        data = (data - np.mean(data)) / np.std(data)
        return data.reshape(1, 42, 1)

    def predict_gesture(self, data):
        print("Original data shape:", data.shape)

        try:
            processed_data = data.reshape(-1, 21, 2)  # تغيير الشكل ليتوافق مع النموذج
        except ValueError as e:
            print(f"Error reshaping data: {e}")
            return None, None

        print("Processed data shape:", processed_data.shape)

        prediction = self.model.predict(processed_data, verbose=0)

        gesture_name = np.argmax(prediction, axis=1)  # قد تحتاج لتغيير هذا حسب منطقك
        confidence = np.max(prediction, axis=1)

        return gesture_name, confidence

    def reset_history(self):
        """
        """
        self.gesture_history.clear()