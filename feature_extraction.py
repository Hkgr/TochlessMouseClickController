# feature_extraction.py
import numpy as np

def extract_features(landmarks):
    features = []
    landmarks = np.array(landmarks)

    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dx = landmarks[j][0] - landmarks[i][0]
            dy = landmarks[j][1] - landmarks[i][1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            angle = np.arctan2(dy, dx)
            features.extend([distance, angle])

    palm_size = np.linalg.norm(landmarks[0] - landmarks[5])
    finger_lengths = []
    for finger_tip in [8, 12, 16, 20]:
        finger_base = finger_tip - 3
        finger_length = np.linalg.norm(landmarks[finger_tip] - landmarks[finger_base])
        finger_lengths.append(finger_length / palm_size)

    features.extend(finger_lengths)

    thumb_finger_angle = np.arctan2(landmarks[8][1] - landmarks[4][1], landmarks[8][0] - landmarks[4][0])
    features.append(thumb_finger_angle)

    center_of_mass = np.mean(landmarks, axis=0)
    features.extend(center_of_mass)

    return features
