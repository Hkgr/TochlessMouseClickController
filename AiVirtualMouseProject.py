import cv2
import numpy as np
import pyautogui
from HandTrackingModule import HandDetector  # استيراد كلاس HandDetector
from tensorflow.keras.models import load_model

# تحميل الموديل المدرب
model = load_model("optimized_gesture_model.h5")  # ضع مسار الموديل هنا
classes = ["0", "1"]
# تهيئة HandDetector
detector = HandDetector(maxHands=1, detectionCon=0.7, trackCon=0.7)

# الحصول على أبعاد الشاشة
screen_width, screen_height = pyautogui.size()

# تهيئة الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # قلب الإطار أفقيًا
    frame = cv2.flip(frame, 1)

    # العثور على اليدين والإحداثيات
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    # تحقق من عدد النقاط في lmList
    print(f"عدد النقاط في lmList: {len(lmList)}")

    if len(lmList) > 0:
        # طباعة النقاط نفسها لتفقد القيم
        print("نقاط lmList:", lmList)

        # استخراج الإحداثيات المسطحة من lmList
        data = np.array([coord[1:] for coord in lmList]).flatten()

        # تحقق من شكل البيانات قبل إرسالها للنموذج
        print(f"بيانات الإدخال قبل إعادة تشكيلها: {data.shape}")

        # تأكد من أن البيانات تحتوي على الحجم الصحيح (في هذه الحالة 210)
        if len(data) == 42:
            # إضافة قيم صفرية لتكملة البيانات إلى 210
            data = np.pad(data, (0, 210 - len(data)), 'constant')
            data = data.reshape(1, 210, 1)  # إعادة تشكيل البيانات لتناسب المدخلات المطلوبة

            print(f"بيانات الإدخال بعد إعادة تشكيلها: {data.shape}")

            # التنبؤ بالإشارة باستخدام الموديل
            prediction = model.predict(data)
            gesture_index = np.argmax(prediction)
            gesture_name = classes[gesture_index]

            # عرض الإشارة المتوقعة على الشاشة
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # تنفيذ التحكم بناءً على الإشارة
            fingers = detector.fingersUp()
            if gesture_name == "1" and fingers[1] == 1 and fingers[2] == 0:
                # تحريك الماوس بناءً على إحداثيات السبابة
                x_mouse = int(lmList[8][1] * screen_width / frame.shape[1])
                y_mouse = int(lmList[8][2] * screen_height / frame.shape[0])
                pyautogui.moveTo(x_mouse, y_mouse)

            elif gesture_name == "3" and fingers[1] == 1 and fingers[2] == 1:
                # تنفيذ النقر المزدوج
                pyautogui.doubleClick()

        else:
            print("البيانات المدخلة غير صحيحة (يجب أن تكون 42 قيمة).")

    else:
        print("لا توجد يد مكتشفة!")

    # عرض الإطار مع النقاط المكتشفة
    cv2.imshow("Gesture Control", frame)

    # إنهاء البرنامج عند الضغط على مفتاح 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
