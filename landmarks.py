import cv2
import numpy as np
import pandas as pd
from HandTrackingModule import HandDetector
import os

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    detector = HandDetector(maxHands=1)
    collected_data = []
    current_gesture = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=True)
        fingers = detector.fingersUp(frame)
        num_fingers_up = fingers.count(1)

        info_text = f"Current Gesture: {current_gesture if current_gesture is not None else 'Not Set'}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples Collected: {len(collected_data)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
       # cv2.putText(frame, f"Fingers Up: {num_fingers_up}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 0-4: Set Gesture | SPACE: Capture | Q: Quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
            current_gesture = int(chr(key))

        elif key == ord(' '):
            if current_gesture is not None and len(lmList) > 0:
                collected_data.append({
                    'Gesture': current_gesture,
                    'Landmarks': str(lmList)
                })

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected_data:
        df = pd.DataFrame(collected_data)
        output_file = "gesture_data.xlsx"
        df.to_excel(output_file, index=False)

if __name__ == "__main__":
    main()
