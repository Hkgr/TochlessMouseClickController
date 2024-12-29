import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from HandTrackingModule import HandDetector
import time

# تحميل النموذج المدرب
model = tf.keras.models.load_model('gesture_model.h5')

# الكاميرا
cap = cv2.VideoCapture(0)

# تعريف المتغيرات لتخزين النتائج
results = []

# تعريف الكاشف
detector = HandDetector()


# دالة لمعالجة الفيديو
def process_video():
    frame_count = 0
    correct_predictions = 0
    total_predictions = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # تحويل الصورة إلى RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # كشف اليد
        lmList = detector.findPosition(img_rgb)

        if lmList:
            # استخراج الميزات (features) من اليد
            features = extract_features(lmList)

            # التنبؤ بالإشارة من النموذج المدرب
            prediction = model.predict(features)
            gesture = np.argmax(prediction)  # الحصول على الإشارة التي حصلت على أعلى احتمالية

            # تحديد الإشارة الصحيحة بناءً على وضع اليد أو البيانات المتاحة
            correct_gesture = get_correct_gesture(img)  # دالة مخصصة للحصول على الإشارة الصحيحة في كل إطار

            # إضافة التنبؤات للمقارنة
            total_predictions += 1
            if gesture == correct_gesture:
                correct_predictions += 1

        # تحديث عرض الصورة
        cv2.imshow("Virtual Mouse", img)

        # إغلاق الكاميرا عند الضغط على 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # حساب الدقة بعد إغلاق الكاميرا
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy:.2f}")

    # حفظ النتائج في جدول بيانات
    results.append({
        'total_frames': frame_count,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'accuracy': accuracy
    })

    # تحويل النتائج إلى DataFrame وحفظها
    df = pd.DataFrame(results)
    df.to_csv("gesture_accuracy.csv", index=False)




# دالة لاستخراج الميزات (حسب المتطلبات)
def extract_features(lmList):
    # فرضاً نأخذ المسافات بين بعض النقاط كنموذج
    features = []
    for i in range(0, len(lmList), 2):
        # حساب الفرق بين النقاط (على سبيل المثال بين x و y)
        dx = lmList[i + 1][1] - lmList[i][1]  # فرق الإحداثيات x
        dy = lmList[i + 1][2] - lmList[i][2]  # فرق الإحداثيات y
        features.append([dx, dy])  # إضافة الفرق بين النقاط
    return np.array(features).reshape(1, -1)


# بدء عملية الفيديو
process_video()

# تحرير الكاميرا
cap.release()
cv2.destroyAllWindows()
