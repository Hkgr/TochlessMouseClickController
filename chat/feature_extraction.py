# feature_extraction.py
import numpy as np


def extract_features(landmarks):
    """
    استخراج الميزات من نقاط اليد
    تحسين: إضافة المزيد من الميزات الهندسية
    """
    features = []
    landmarks = np.array(landmarks)

    # المسافات بين النقاط
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dx = landmarks[j][0] - landmarks[i][0]
            dy = landmarks[j][1] - landmarks[i][1]
            # المسافة الإقليدية
            distance = np.sqrt(dx ** 2 + dy ** 2)
            # الزاوية
            angle = np.arctan2(dy, dx)
            features.extend([distance, angle])

    # إضافة ميزات النسب
    palm_size = np.linalg.norm(landmarks[0] - landmarks[5])
    finger_lengths = []
    for finger_tip in [8, 12, 16, 20]:
        finger_base = finger_tip - 3
        finger_length = np.linalg.norm(landmarks[finger_tip] - landmarks[finger_base])
        finger_lengths.append(finger_length / palm_size)

    features.extend(finger_lengths)
    return features