# main.py
import numpy as np
import pandas as pd
from feature_extraction import extract_features


# دالة لتحويل عمود Landmarks إلى قائمة إحداثيات
def parse_landmarks(landmarks_str):
    # إزالة الأقواس المزدوجة وتحويل النص إلى قائمة حقيقية
    landmarks = landmarks_str.strip("[]").split("], [")
    landmarks = [list(map(int, point.split(", "))) for point in landmarks]
    return landmarks


# قراءة البيانات من ملف Excel
file_path = "gesture_data.xlsx"
data = pd.read_excel(file_path)

# قائمة لتخزين البيانات المجهزة
features_list = []
labels = []

for index, row in data.iterrows():
    # استخراج الإشارة (الـ Gesture)
    gesture = row['Gesture']
    # استخراج إحداثيات الـ Landmarks
    landmarks = parse_landmarks(row['Landmarks'])

    # استخراج الميزات من الـ Landmarks باستخدام الدالة
    features = extract_features(landmarks)

    # إضافة الميزات والـ Label (الإشارة) إلى القوائم
    features_list.append(features)
    labels.append(gesture)

# تحويل الميزات إلى مصفوفة Numpy
features_array = np.array(features_list)
labels_array = np.array(labels)

# عرض أول 5 ميزات كاختبار
print(features_array[:5])

# حفظ البيانات في ملف Excel جديد
features_df = pd.DataFrame(features_array)
features_df['Gesture'] = labels_array
features_df.to_excel("gesture_features.xlsx", index=False)

print("تم استخراج الميزات وحفظها في gesture_features.xlsx")
