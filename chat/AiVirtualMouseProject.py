import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time


class RealTimeGestureRecognizer:
    def __init__(self, model_path, confidence_threshold=0.7):
        self.model = load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.prediction_history = []
        self.history_length = 5
        self.gesture_names = {
            0: "Zero - قبضة",
            1: "One - سبابة",
            2: "Two - سبابة ووسطى",
            3: "Three - ثلاث أصابع",
            4: "Four - أربع أصابع"
        }

    def extract_additional_features(self, landmarks):
        features = []
        landmarks_array = np.array(landmarks).reshape(-1, 2)
        for i in range(len(landmarks_array)):
            for j in range(i + 1, len(landmarks_array)):
                dist = np.linalg.norm(landmarks_array[i] - landmarks_array[j])
                features.append(dist)

        for i in range(len(landmarks_array) - 2):
            v1 = landmarks_array[i + 1] - landmarks_array[i]
            v2 = landmarks_array[i + 2] - landmarks_array[i + 1]
            angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
            features.append(angle)
        return features

    def preprocess_landmarks(self, hand_landmarks, frame_shape):
        basic_landmarks = []
        for landmark in hand_landmarks.landmark:
            x = landmark.x * frame_shape[1]
            y = landmark.y * frame_shape[0]
            basic_landmarks.extend([x, y])

        additional_features = self.extract_additional_features(
            [basic_landmarks[i:i + 2] for i in range(0, len(basic_landmarks), 2)]
        )

        all_features = np.array(basic_landmarks + additional_features, dtype=np.float32)
        all_features = (all_features - np.mean(all_features)) / np.std(all_features)
        padded_features = np.zeros(424)
        padded_features[:len(all_features)] = all_features
        return padded_features.reshape(1, 424, 1)

    def get_smoothed_prediction(self, prediction):
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > self.history_length:
            self.prediction_history.pop(0)
        avg_prediction = np.mean(self.prediction_history, axis=0)
        return np.argmax(avg_prediction), np.max(avg_prediction)

    def draw_hand_info(self, frame, gesture_id, confidence):
        # تحديد اسم الإيماءة
        gesture_name = self.gesture_names.get(gesture_id, "Unknown")

        # تحضير النصوص
        gesture_text = f"GESTURE #{gesture_id}"
        name_text = f"{gesture_name}"
        confidence_text = f"Confidence: {confidence:.2f}"

        # تحديد المواقع والألوان
        bg_color = (0, 0, 0)
        text_color = (255, 255, 255)

        # رسم خلفية سوداء شفافة
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), bg_color, -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # رسم النصوص
        cv2.putText(frame, gesture_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
        cv2.putText(frame, name_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        cv2.putText(frame, confidence_text, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera not found!")
            return

        print("=== بدء التعرف على الإيماءات ===")
        while True:
            success, frame = cap.read()
            if not success:
                print("خطأ في قراءة الإطار!")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            print("\n--- تحليل الإطار الجديد ---")
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    try:
                        landmarks = self.preprocess_landmarks(hand_landmarks, frame.shape)
                        prediction = self.model.predict(landmarks, verbose=0)
                        gesture_id, confidence = self.get_smoothed_prediction(prediction[0])

                        print(f"الإيماءة المكتشفة: {self.gesture_names[gesture_id]}")
                        print(f"مستوى الثقة: {confidence:.2f}")

                        if confidence > self.confidence_threshold:
                            print("✓ تم تأكيد الإيماءة")
                        else:
                            print("× ثقة منخفضة - تم تجاهل الإيماءة")

                    except Exception as e:
                        print(f"خطأ في المعالجة: {str(e)}")
            else:
                print("لم يتم اكتشاف يد")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap = cv2.VideoCapture(0)
        print("Starting...")

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    print("Hand detected!")

            # Display frame
            cv2.imshow("Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = RealTimeGestureRecognizer(
        model_path="gesture_model.h5",
        confidence_threshold=0.6
    )
    recognizer.run()