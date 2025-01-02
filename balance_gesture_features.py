# balance_gesture_features.py

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os


def load_data(file_path):
    print(f"جارٍ تحميل البيانات من الملف: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"الملف {file_path} غير موجود.")

    data = pd.read_excel(file_path)

    if 'Gesture' not in data.columns:
        raise ValueError("العمود 'Gesture' غير موجود في البيانات.")

    print(f"تم تحميل البيانات بنجاح. عدد الأسطر: {len(data)}")
    return data


def check_balance(data):
    gesture_counts = data['Gesture'].value_counts(normalize=True) * 100
    print("توزيع الفئات قبل الموازنة (بالنسب المئوية):")
    print(gesture_counts)
    return gesture_counts


#  SMOTE
def balance_data(data):
    print("جارٍ تطبيق SMOTE لموازنة البيانات...")

    X = data.drop(columns=['Gesture']).values
    y = data['Gesture']

    print("جارٍ تقسيم البيانات إلى مجموعات تدريب واختبار...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"تم تقسيم البيانات بنجاح. عدد بيانات التدريب: {len(X_train)}, عدد بيانات الاختبار: {len(X_test)}")

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    print("جارٍ تطبيق SMOTE على مجموعة التدريب...")
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"تم تطبيق SMOTE بنجاح. عدد البيانات بعد الموازنة: {len(X_resampled)}")

    feature_columns = [f"Feature_{i + 1}" for i in range(X_resampled.shape[1])]
    balanced_data = pd.DataFrame(X_resampled, columns=feature_columns)
    balanced_data['Gesture'] = y_resampled

    print("توزيع الفئات بعد الموازنة:")
    check_balance(balanced_data)

    return balanced_data, X_test, y_test


def save_data(data, file_path, csv_path=None):
    print(f"جارٍ حفظ البيانات في الملف: {file_path}")
    data.to_excel(file_path, index=False)
    print(f"تم حفظ البيانات في {file_path}")

    # if csv_path:
    #     print(f"جارٍ حفظ البيانات في الملف: {csv_path}")
    #     data.to_csv(csv_path, index=False)
    #     print(f"تم حفظ البيانات في {csv_path}")


if __name__ == "__main__":
    input_file = "gesture_features.xlsx"
    output_file = "balanced_gesture_features.xlsx"
    output_csv_file = "balanced_gesture_features.csv"

    try:
        print("بداية معالجة البيانات...")
        data = load_data(input_file)
        print("قبل الموازنة:")
        check_balance(data)

        balanced_data, X_test, y_test = balance_data(data)
        save_data(balanced_data, output_file, output_csv_file)

        print("تمت عملية الموازنة بنجاح.")
    except Exception as e:
        print(f"حدث خطأ: {str(e)}")
