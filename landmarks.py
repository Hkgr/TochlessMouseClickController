import cv2
import openpyxl
from HandTrackingModule import HandDetector  # استدعاء الكاشف من وحدة HandTrackingModule
import os

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# إنشاء كائن من HandDetector
detector = HandDetector()

# اسم ملف Excel
file_name = "gesture_data.xlsx"

# التحقق مما إذا كان الملف موجودًا
if os.path.exists(file_name):
    wb = openpyxl.load_workbook(file_name)  # فتح الملف الموجود
    ws = wb.active  # استخدام الورقة الحالية
else:
    wb = openpyxl.Workbook()  # إنشاء ملف جديد
    ws = wb.active
    ws.title = "Gestures Data"
    ws.append(['Gesture', 'Landmarks'])  # إضافة رأس الأعمدة

# متغير لتتبع ما إذا كان يجب جمع البيانات
capture_data = False
landmarks_list = []  # لتخزين الإحداثيات المؤقتة حتى يتم إدخال الإشارة

while True:
    ret, img = cap.read()
    if not ret:
        break

    # استخدام الكاشف لاكتشاف اليدين
    img = detector.findHands(img)  # العثور على اليدين
    lmList = detector.findPosition(img)  # استخراج المعالم (landmarks)

    if lmList:
        # جمع الإحداثيات لكل نقطة (landmark)
        landmark_list = []
        for lm in lmList:
            landmark_list.append([lm[1], lm[2]])  # فقط x و y الإحداثيات

        # إذا تم تفعيل جمع البيانات عند الضغط على Enter
        if capture_data:
            # تخزين البيانات المؤقتة
            landmarks_list.append(landmark_list)
            print("تم تسجيل البيانات لهذا الإطار.")
            capture_data = False  # إيقاف جمع البيانات بعد التسجيل

    # عرض الصورة في نافذة
    cv2.imshow("Capture Gestures", img)

    # الانتظار لضغط زر Enter لجمع البيانات لهذا الإطار
    if cv2.waitKey(1) & 0xFF == 13:  # الكود 13 هو زر Enter
        capture_data = True
        print("تم تفعيل جمع البيانات، اضغط Enter لتسجيل الإشارة لهذا الإطار...")

    # عند الضغط على 'q' للخروج من البرنامج
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# الآن بعد أن جمعنا البيانات (landmarks) لعدة إطارات، يمكن إدخال الإشارة يدويًا
print("العملية انتهت. الآن يمكنك إدخال الإشارات يدويًا في ملف Excel.")

# إدخال الإشارات يدويًا لكل إطار تم جمعه
for i, landmarks in enumerate(landmarks_list):
    gesture = input(f"أدخل الإشارة للإطار {i + 1} (مثل 'سبابة مرفوعة'): ")
    ws.append([gesture, str(landmarks)])

# حفظ ملف Excel
wb.save(file_name)

cap.release()
cv2.destroyAllWindows()
