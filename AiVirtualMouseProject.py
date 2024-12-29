import cv2
import HandTrackingModule as htm
import pyautogui
import time

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
        index_finger_tip = lmList[8]
        middle_finger_tip = lmList[12]

        x = index_finger_tip[1]
        y = index_finger_tip[2]
        screen_x = int(screen_width * (x / img.shape[1]))
        screen_y = int(screen_height * (y / img.shape[0]))

        pyautogui.moveTo(screen_x, screen_y)

        fingers = detector.fingersUp()

        print(f"Fingers: {fingers}")

        if fingers[1] == 1 and fingers[2] == 1:
            current_time = time.time()
            if current_time - last_click_time > 0.2:
                pyautogui.doubleClick()
                last_click_time = current_time
                print("Double Click Detected!")
            else:
                print("Double Click Not Detected - Time Interval Too Short")
        else:
            print("Index and Middle Finger Not Both Raised")

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
