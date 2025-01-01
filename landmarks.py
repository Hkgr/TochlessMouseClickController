import cv2
import numpy as np
import pandas as pd
from HandTrackingModule import HandDetector
import os


def main():
    # تهيئة الكاميرا
    print("جاري تهيئة الكاميرا...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("خطأ: لم يتم العثور على الكاميرا!")
        return

    # تهيئة كاشف اليد
    print("جاري تهيئة كاشف اليد...")
    detector = HandDetector(maxHands=1)

    # قائمة لتخزين البيانات
    collected_data = []
    current_gesture = None

    print("\nتعليمات الاستخدام:")
    print("1. اضغط على مفتاح من 0 إلى 4 لتحديد رقم الإيماءة")
    print("2. اضغط SPACE لالتقاط الإيماءة الحالية")
    print("3. اضغط q للخروج وحفظ البيانات")
    print("\nجاهز للبدء!")

    while True:
        # قراءة الإطار من الكاميرا
        success, frame = cap.read()
        if not success:
            print("خطأ في قراءة الإطار من الكاميرا!")
            break

        # قلب الإطار أفقياً لتسهيل الاستخدام
        frame = cv2.flip(frame, 1)

        # تحديد موقع اليد
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=True)

        # الحصول على حالة الأصابع
        fingers = detector.fingersUp(frame)
        # حساب عدد الأصابع المرفوعة
        num_fingers_up = fingers.count(1)

        # عرض المعلومات على الشاشة
        info_text = f"Current Gesture: {current_gesture if current_gesture is not None else 'Not Set'}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples Collected: {len(collected_data)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Fingers Up: {num_fingers_up}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 0-4: Set Gesture | SPACE: Capture | Q: Quit",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # عرض الإطار
        cv2.imshow("Data Collection", frame)

        # التحكم في المفاتيح
        key = cv2.waitKey(1) & 0xFF

        # تحديد الإيماءة (0-4)
        if key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
            current_gesture = int(chr(key))
            print(f"\nتم تحديد الإيماءة: {current_gesture}")

        # التقاط البيانات (مفتاح المسافة)
        elif key == ord(' '):
            if current_gesture is not None and len(lmList) > 0:
                collected_data.append({
                    'Gesture': current_gesture,
                    'Landmarks': str(lmList)
                })
                print(f"تم التقاط الإيماءة {current_gesture}. العدد الإجمالي: {len(collected_data)}")
            else:
                print("لا يمكن التقاط البيانات. تأكد من تحديد الإيماءة وظهور اليد في الكاميرا!")

        # الخروج (q)
        elif key == ord('q'):
            break

    # إغلاق الكاميرا
    cap.release()
    cv2.destroyAllWindows()

    # حفظ البيانات
    if collected_data:
        df = pd.DataFrame(collected_data)
        output_file = "gesture_data.xlsx"
        df.to_excel(output_file, index=False)
        print(f"\nتم حفظ {len(collected_data)} عينة في الملف {output_file}")
    else:
        print("\nلم يتم جمع أي بيانات!")


if __name__ == "__main__":
    main()
