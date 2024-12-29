import cv2
import HandTrackingModule as htm
import pyautogui
import time
from tensorflow.keras.models import load_model
import numpy as np

# تحميل النموذج المدرب
model = load_model("gesture_model.h5")

cap = cv2.VideoCapture(0)
detector = htm.HandDetector(detectionCon=0.7)

screen_width, screen_height = pyautogui.size()

last_click_time = 0
prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        # استخراج الميزات من نقاط اليد
        features = [lm[1:] for lm in lmList[:20]]  # خذ أول 20 نقطة فقط كميزة
        features = np.array(features).flatten().reshape(1, -1)

        # التنبؤ بالإيماءة باستخدام النموذج
        gesture = (model.predict(features) > 0.5).astype("int32")

        if gesture == 1:  # إذا كان الإيماءة تعني النقر
            pyautogui.click()

        # متابعة المؤشر
        index_finger_tip = lmList[8]
        x, y = index_finger_tip[1], index_finger_tip[2]
        screen_x = int(screen_width * (x / img.shape[1]))
        screen_y = int(screen_height * (y / img.shape[0]))
        pyautogui.moveTo(screen_x, screen_y)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
