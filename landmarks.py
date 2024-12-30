import cv2
import openpyxl
from HandTrackingModule import HandDetector
import os

cap = cv2.VideoCapture(0)
detector = HandDetector()

file_name = "gesture_data.xlsx"

if os.path.exists(file_name):
    wb = openpyxl.load_workbook(file_name)
    ws = wb.active
else:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Gestures Data"
    ws.append(['Gesture', 'Landmarks'])

capture_data = False
landmarks_list = []

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if lmList:
        landmark_list = []
        for lm in lmList:
            landmark_list.append([lm[1], lm[2]])

        if capture_data:
            landmarks_list.append(landmark_list)
            print("تم تسجيل البيانات لهذا الإطار.")
            capture_data = False

    cv2.imshow("Capture Gestures", img)

    if cv2.waitKey(1) & 0xFF == 13:
        capture_data = True
        print("تم تفعيل جمع البيانات، اضغط Enter لتسجيل الإشارة لهذا الإطار...")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("العملية انتهت. الآن يمكنك إدخال الإشارات يدويًا في ملف Excel.")

for i, landmarks in enumerate(landmarks_list):
    gesture = input(f"أدخل الإشارة للإطار {i + 1} (مثل 'سبابة مرفوعة'): ")
    ws.append([gesture, str(landmarks)])

wb.save(file_name)

cap.release()
cv2.destroyAllWindows()
