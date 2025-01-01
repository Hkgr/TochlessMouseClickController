# gesture_features.py

import numpy as np
import pandas as pd
from feature_extraction import extract_features


def parse_landmarks(landmarks_str):
    landmarks = landmarks_str.strip("[]").split("], [")
    landmarks = [list(map(int, point.split(", "))) for point in landmarks]
    return landmarks


file_path = "gesture_data.xlsx"
data = pd.read_excel(file_path)

features_list = []
labels = []

try:
    for index, row in data.iterrows():
        gesture = row['Gesture']
        landmarks = parse_landmarks(row['Landmarks'])

        # التحقق من صحة البيانات
        if landmarks and all(len(point) == 3 for point in landmarks):  # التحقق من أن كل نقطة تحتوي على 3 عناصر
            features = extract_features(landmarks)
            features_list.append(features)
            labels.append(gesture)
        else:
            print(f"البيانات غير صحيحة في السطر {index}")
except Exception as e:
    print(f"حدث خطأ في معالجة السطر {index}: {str(e)}")


features_array = np.array(features_list)
labels_array = np.array(labels)

# تطبيع الميزات إذا لزم الأمر (اختياري)
# features_array = (features_array - features_array.mean(axis=0)) / features_array.std(axis=0)

features_df = pd.DataFrame(features_array)
features_df['Gesture'] = labels_array
features_df.to_excel("gesture_features.xlsx", index=False)

print("تم استخراج الميزات وحفظها في gesture_features.xlsx")
