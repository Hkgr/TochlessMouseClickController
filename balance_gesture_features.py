# balance_gesture_features.py

import pandas as pd
from imblearn.over_sampling import SMOTE

# ------------------------
# 1. تحميل البيانات
# ------------------------
def load_data(file_path):
    """
    تحميل البيانات من ملف Excel.
    """
    data = pd.read_excel(file_path)
    return data

# ------------------------
# 2. التحقق من توازن الفئات
# ------------------------
def check_balance(data):
    """
    التحقق من توزيع الفئات.
    """
    gesture_counts = data['Gesture'].value_counts()
    print("توزيع الفئات:")
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

    # تطبيق SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # تحويل البيانات المعالجة إلى DataFrame جديد
    feature_columns = [f"Feature_{i+1}" for i in range(X_resampled.shape[1])]
    balanced_data = pd.DataFrame(X_resampled, columns=feature_columns)
    balanced_data['Gesture'] = y_resampled
    return balanced_data

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

    # تحميل البيانات
    data = load_data(input_file)

    # التحقق من توازن الفئات قبل التوزيع
    print("قبل التوزيع:")
    check_balance(data)

    # تطبيق إعادة التوزيع
    balanced_data = balance_data(data)

    # التحقق من توازن الفئات بعد التوزيع
    print("بعد التوزيع:")
    check_balance(balanced_data)

    # حفظ البيانات المتوازنة
    save_data(balanced_data, output_file)