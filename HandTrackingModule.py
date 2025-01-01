# HandTrackingModule.py
import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = float(detectionCon)
        self.trackCon = float(trackCon)

        # تهيئة MediaPipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # نقاط أطراف الأصابع

    def findHands(self, img, draw=True):
        """
        اكتشاف اليدين في الصورة
        """
        if img is None:
            return None

        # تحويل الصورة إلى RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        تحديد مواقع نقاط اليد
        """
        lmList = []
        if img is None:
            return lmList

        if self.results.multi_hand_landmarks:
            try:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            except IndexError:
                pass
        return lmList

    def fingersUp(self, img):
        """
        تحديد حالة الأصابع (مرفوعة أم لا)
        """
        # الحصول على مواقع اليد
        lmList = self.findPosition(img, draw=False)  # استخدم findPosition للحصول على lmList
        fingers = []

        if len(lmList) == 0:
            return [0, 0, 0, 0, 0]  # في حالة عدم وجود اليد

        # الإبهام
        if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # باقي الأصابع
        for id in range(1, 5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
