import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from HandTrackingModule import HandDetector
from sklearn.preprocessing import StandardScaler
import pyautogui
import math


def map_coordinates(x, y, frame_rect, screen_size):
    """
    Convert hand coordinates to screen coordinates
    """
    in_min_x, in_min_y = frame_rect[0], frame_rect[1]
    in_max_x, in_max_y = frame_rect[2], frame_rect[3]
    out_min_x, out_min_y = 0, 0
    out_max_x, out_max_y = screen_size[0], screen_size[1]

    x = (x - in_min_x) * (out_max_x - out_min_x) / (in_max_x - in_min_x) + out_min_x
    y = (y - in_min_y) * (out_max_y - out_min_y) / (in_max_y - in_min_y) + out_min_y

    return int(x), int(y)


def smooth_move(x, y, current_x, current_y, smoothing=0.5):
    """
    Smooth pointer movement
    """
    return int(current_x + (x - current_x) * smoothing), int(current_y + (y - current_y) * smoothing)


def extract_features(landmark_list, fingers_state):
    # [Previous extract_features implementation remains the same]
    pass


def create_transparent_overlay(image, alpha=0.3):
    """
    Create a semi-transparent overlay for text
    """
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (400, 250), (40, 40, 40), -1)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def main():
    detector = HandDetector(maxHands=1, detectionCon=0.7)

    # Set larger window size
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    pyautogui.FAILSAFE = False

    # Detection zone (scaled for larger window)
    FRAME_RECT = (200, 150, 1080, 570)

    # Tracking states
    prev_fingers = [0, 0, 0, 0, 0]
    current_x, current_y = screen_width // 2, screen_height // 2

    # Define gestures
    gestures = {
        "MOVE": "Moving Cursor",
        "CLICK": "Single Click",
        "DOUBLE_CLICK": "Double Click"
    }

    # Colors
    TEXT_COLOR = (0, 255, 155)
    HIGHLIGHT_COLOR = (255, 191, 0)
    FRAME_COLOR = (0, 255, 155)

    # Fonts
    MAIN_FONT = cv2.FONT_HERSHEY_SIMPLEX
    STATUS_FONT_SCALE = 1.2
    INFO_FONT_SCALE = 0.8

    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read camera")
            break

        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=True)

        # Calculate FPS
        cTime = time.time()
        fps = int(1 / (cTime - pTime))
        pTime = cTime

        # Create transparent overlay for information
        img = create_transparent_overlay(img)

        if len(lmList) > 0:
            try:
                # Get finger states
                fingers = detector.fingersUp(img)

                # Get index finger position
                index_finger = lmList[8]

                # Initialize current gesture
                current_gesture = "No Gesture"

                # Index finger only - Move cursor
                if fingers[1] == 1 and sum(fingers[2:]) == 0:
                    x, y = map_coordinates(index_finger[1], index_finger[2], FRAME_RECT, (screen_width, screen_height))
                    current_x, current_y = smooth_move(x, y, current_x, current_y)
                    pyautogui.moveTo(current_x, current_y)
                    current_gesture = gestures["MOVE"]

                # Index and middle fingers - Single click
                elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0:
                    if prev_fingers[1] == 1 and prev_fingers[2] == 1:
                        pyautogui.click()
                        time.sleep(0.1)
                    current_gesture = gestures["CLICK"]

                # Index, middle, and ring fingers - Double click
                elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    if prev_fingers[1] == 1 and prev_fingers[2] == 1 and prev_fingers[3] == 1:
                        pyautogui.doubleClick()
                        time.sleep(0.1)
                    current_gesture = gestures["DOUBLE_CLICK"]

                # Display status information
                cv2.putText(img, f'FPS: {fps}', (20, 50),
                            MAIN_FONT, STATUS_FONT_SCALE, TEXT_COLOR, 2)
                cv2.putText(img, f'Current Gesture: {current_gesture}', (20, 100),
                            MAIN_FONT, STATUS_FONT_SCALE, HIGHLIGHT_COLOR, 2)

                # Update previous finger states
                prev_fingers = fingers.copy()

            except Exception as e:
                print(f"Prediction error: {str(e)}")

        # Add detection zone
        cv2.rectangle(img, FRAME_RECT[:2], FRAME_RECT[2:], FRAME_COLOR, 3)
        cv2.putText(img, "Hand Detection Zone", (FRAME_RECT[0], FRAME_RECT[1] - 20),
                    MAIN_FONT, 1, FRAME_COLOR, 2)

        # Add instructions
        instructions = [
            "GESTURE CONTROL INSTRUCTIONS:",
            "1. Index finger only → Move cursor",
            "2. Index + Middle fingers → Single click",
            "3. Index + Middle + Ring fingers → Double click",
            "Press 'q' to exit"
        ]

        y_offset = img.shape[0] - 200
        for i, instruction in enumerate(instructions):
            if i == 0:  # Title
                cv2.putText(img, instruction, (20, y_offset + i * 35),
                            MAIN_FONT, INFO_FONT_SCALE, HIGHLIGHT_COLOR, 2)
            else:  # Instructions
                cv2.putText(img, instruction, (20, y_offset + i * 35),
                            MAIN_FONT, INFO_FONT_SCALE, TEXT_COLOR, 2)

        # Show window with custom title
        cv2.namedWindow("Gesture Mouse Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture Mouse Control", 1280, 720)
        cv2.imshow("Gesture Mouse Control", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()