import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# تحميل النموذج المدرب
model = load_model("optimized_gesture_model.h5")

# إعداد Mediapipe لتتبع اليد
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# قائمة الإيماءات الممكنة (محدثة لخمسة إيماءات)
classes = ["No Gesture", "Move Mouse", "Double Click", "Scroll Up", "Scroll Down"]

# فتح كاميرا الويب
cap = cv2.VideoCapture(0)

while True:
    # قراءة إطار من الكاميرا
    ret, frame = cap.read()

    if not ret:
        print("فشل في التقاط الإطار")
        break

    # تحويل الصورة إلى RGB لأن Mediapipe يعمل على هذا التنسيق
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # اكتشاف اليدين
    result = hands.process(frame_rgb)

    # إذا تم اكتشاف يد
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # رسم المعالم على اليد
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # استخراج إحداثيات النقاط (إزالة الفهرس الأول لأنه لا يحتوي على معلومات الموقع)
            lmList = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            # إذا كان عدد النقاط أقل من 21، نعرض رسالة
            if len(lmList) != 21:
                print("عدد النقاط في lmList:", len(lmList))
                continue

            # تحويل إحداثيات اليد إلى بيانات الإدخال
            data = np.array([coord[0] for coord in lmList] +
                            [coord[1] for coord in lmList] +
                            [coord[2] for coord in lmList]).flatten().reshape(1, 21,
                                                                              3)  # يجب أن يكون حجم البيانات (1, 21, 3)

            # إعادة تشكيل البيانات لتتناسب مع متطلبات النموذج
            data = data.reshape(1, 210, 1)  # إعادة تشكيل البيانات إلى (1, 210, 1)

            # التنبؤ بالإشارة باستخدام الموديل
            prediction = model.predict(data)

            # الحصول على الفهرس الأعلى في التنبؤ
            gesture_index = np.argmax(prediction)

            # ضمان أن الفهرس ضمن النطاق المناسب
            gesture_index = max(0, min(gesture_index, len(classes) - 1))  # ضمان أن الفهرس ضمن النطاق

            # الحصول على اسم الإشارة
            gesture_name = classes[gesture_index]

            # عرض الإشارة المتوقعة على الشاشة
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # عرض الإطار
    cv2.imshow("Hand Gesture Recognition", frame)

    # الخروج عند الضغط على مفتاح 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تحرير الكاميرا وإغلاق النوافذ
cap.release()
cv2.destroyAllWindows()
