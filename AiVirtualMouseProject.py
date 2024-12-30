import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from HandTrackingModule import HandDetector
import time

model = tf.keras.models.load_model('gesture_model.h5')
cap = cv2.VideoCapture(0)
results = []
detector = HandDetector()

def process_video():
    frame_count = 0
    correct_predictions = 0
    total_predictions = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lmList = detector.findPosition(img_rgb)

        if lmList:
            features = extract_features(lmList)
            prediction = model.predict(features)
            gesture = np.argmax(prediction)
            correct_gesture = get_correct_gesture(img)

            total_predictions += 1
            if gesture == correct_gesture:
                correct_predictions += 1

        cv2.imshow("Virtual Mouse", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy:.2f}")

    results.append({
        'total_frames': frame_count,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'accuracy': accuracy
    })

    df = pd.DataFrame(results)
    df.to_csv("gesture_accuracy.csv", index=False)

def extract_features(lmList):
    features = []
    for i in range(0, len(lmList), 2):
        dx = lmList[i + 1][1] - lmList[i][1]
        dy = lmList[i + 1][2] - lmList[i][2]
        features.append([dx, dy])
    return np.array(features).reshape(1, -1)

process_video()

cap.release()
cv2.destroyAllWindows()
