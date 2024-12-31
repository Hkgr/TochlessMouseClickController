#balance_gesture_featuers.py

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os


# ------------------------
# 1. تحميل البيانات
# ------------------------
def load_data(file_path):
    """
    تحميل البيانات من ملف Excel.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"الملف {file_path} غير موجود.")

    data = pd.read_excel(file_path)
    return data


# ------------------------
# 2. التحقق من توازن الفئات
# ------------------------
def check_balance(data):
    """
    التحقق من توزيع الفئات.
    """
    gesture_counts = data['Gesture'].value_counts(normalize=True) * 100  # عرض النسب المئوية
    print("توزيع الفئات (بالنسب المئوية):")
    print(gesture_counts)
    return gesture_counts


# ------------------------
# 3. تطبيق إعادة التوزيع (SMOTE)
# ------------------------
def balance_data(data):
    """
    استخدام SMOTE لإعادة توازن البيانات.
    """
    # ميزات التدريب
    X = data.drop(columns=['Gesture']).values
    y = data['Gesture']

    # تقسيم البيانات إلى مجموعة تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # تطبيق SMOTE فقط على مجموعة التدريب
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # تحويل البيانات المعالجة إلى DataFrame جديد
    feature_columns = [f"Feature_{i + 1}" for i in range(X_resampled.shape[1])]
    balanced_data = pd.DataFrame(X_resampled, columns=feature_columns)
    balanced_data['Gesture'] = y_resampled

    # عرض التوزيع بعد الموازنة
    print("توزيع الفئات بعد الموازنة:")
    check_balance(balanced_data)

    # إعادة البيانات المتوازنة
    return balanced_data, X_test, y_test


# ------------------------
# 4. حفظ البيانات المتوازنة
# ------------------------
def save_data(data, file_path):
    """
    حفظ البيانات المتوازنة إلى ملف Excel.
    """
    data.to_excel(file_path, index=False)
    print(f"تم حفظ البيانات في {file_path}")


# ------------------------
# 5. البرنامج الرئيسي
# ------------------------
if __name__ == "__main__":
    # اسم ملف الإدخال
    input_file = "gesture_features.xlsx"
    # اسم ملف الإخراج
    output_file = "balanced_gesture_features.xlsx"

    try:
        # تحميل البيانات
        data = load_data(input_file)

        # التحقق من توازن الفئات قبل التوزيع
        print("قبل التوزيع:")
        check_balance(data)

        # تطبيق إعادة التوزيع
        balanced_data, X_test, y_test = balance_data(data)

        # حفظ البيانات المتوازنة
        save_data(balanced_data, output_file)

        # التحقق من التوازن بعد الموازنة
        print("تمت عملية الموازنة بنجاح.")
        print("توزيع الفئات بعد الموازنة:")
        check_balance(balanced_data)

    except Exception as e:
        print(f"حدث خطأ: {str(e)}")
