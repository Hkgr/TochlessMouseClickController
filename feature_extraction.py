#gesture_extraction

import numpy as np

def extract_features(landmarks):
    features = []
    landmarks = np.array(landmarks)

    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dx = landmarks[j][0] - landmarks[i][0]
            dy = landmarks[j][1] - landmarks[i][1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            features.append(distance)

    return features
