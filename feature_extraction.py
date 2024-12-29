# feature_extraction.py

import numpy as np


def extract_features(landmarks):
    """
    استخراج الميزات من بيانات الـ Landmarks.
    - البيانات هي قائمة من الإحداثيات (x, y) لكل نقطة يد.
    """
    features = []
    # تحويل بيانات الـ Landmarks إلى مصفوفة من إحداثيات (x, y)
    landmarks = np.array(landmarks)

    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            # حساب المسافة بين النقاط (x1, y1) و (x2, y2)
            dx = landmarks[j][0] - landmarks[i][0]  # x
            dy = landmarks[j][1] - landmarks[i][1]  # y
            distance = np.sqrt(dx ** 2 + dy ** 2)  # حساب المسافة بين النقاط في 2D
            features.append(distance)

    return features
