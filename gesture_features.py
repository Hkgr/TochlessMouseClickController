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

for index, row in data.iterrows():
    gesture = row['Gesture']
    landmarks = parse_landmarks(row['Landmarks'])
    features = extract_features(landmarks)
    features_list.append(features)
    labels.append(gesture)

features_array = np.array(features_list)
labels_array = np.array(labels)

print(features_array[:5])

features_df = pd.DataFrame(features_array)
features_df['Gesture'] = labels_array
features_df.to_excel("gesture_features.xlsx", index=False)

print("تم استخراج الميزات وحفظها في gesture_features.xlsx")
