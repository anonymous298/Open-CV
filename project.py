import cv2
import mediapipe as mp
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import numpy as np
import pyautogui
import keyboard
from math import hypot

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        return img

    def find_position(self, img):
        landmark_lists = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                hand_landmarks_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_landmarks_list.append([id, cx, cy])
                landmark_lists.append(hand_landmarks_list)
        return landmark_lists

    def count_fingers(self, landmark_list):
        if len(landmark_list) == 0:
            return 0

        fingers = []
        # Thumb - checking if thumb tip (4) is right of thumb base (2) for right hand
        if landmark_list[4][1] > landmark_list[2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers - checking if fingertip is above finger pip joint
        for tip in [8, 12, 16, 20]:
            if landmark_list[tip][2] < landmark_list[tip - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    # Initialize audio control
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol, maxVol = volRange[0], volRange[1]
    
    # Screen dimensions
    screen_width, screen_height = pyautogui.size()
    
    # Smoothing factor for mouse movement
    smoothening = 5
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0
    
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        img = detector.find_hands(img)
        landmark_lists = detector.find_position(img)
        
        for landmark_list in landmark_lists:
            if len(landmark_list) != 0:
                finger_count = detector.count_fingers(landmark_list)
                
                # Get the positions of thumb and index finger
                if len(landmark_list) >= 9:
                    thumb_tip = landmark_list[4]
                    index_tip = landmark_list[8]
                    middle_tip = landmark_list[12]
                    
                    # Calculate distance between thumb and index finger
                    length = hypot(index_tip[1] - thumb_tip[1], index_tip[2] - thumb_tip[2])
                    
                    # Convert ranges
                    vol = np.interp(length, [30, 300], [minVol, maxVol])
                    brightness = np.interp(length, [30, 300], [0, 100])
                    
                    # Different controls based on finger count
                    if finger_count == 2:  # Volume control
                        volume.SetMasterVolumeLevel(vol, None)
                        cv2.putText(img, f'Volume: {int(np.interp(vol, [minVol, maxVol], [0, 100]))}%', 
                                  (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                    
                    elif finger_count == 3:  # Brightness control
                        sbc.set_brightness(int(brightness))
                        cv2.putText(img, f'Brightness: {int(brightness)}%', 
                                  (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    
                    elif finger_count == 1:  # Mouse control mode
                        # Convert coordinates
                        x3 = np.interp(index_tip[1], [0, w], [0, screen_width])
                        y3 = np.interp(index_tip[2], [0, h], [0, screen_height])
                        
                        # Smooth values
                        curr_x = prev_x + (x3 - prev_x) / smoothening
                        curr_y = prev_y + (y3 - prev_y) / smoothening
                        
                        # Move mouse
                        pyautogui.moveTo(curr_x, curr_y)
                        cv2.circle(img, (index_tip[1], index_tip[2]), 15, (255, 0, 255), cv2.FILLED)
                        prev_x, prev_y = curr_x, curr_y
                        
                        # Click if thumb and index are close
                        if length < 40:  # If fingers are close enough, click
                            pyautogui.click()
                            cv2.circle(img, (index_tip[1], index_tip[2]), 15, (0, 255, 0), cv2.FILLED)
                    
                    elif finger_count == 4:  # Media control
                        if thumb_tip[1] < index_tip[1]:  # Thumb left of index
                            keyboard.press('left')  # Previous track/rewind
                            cv2.putText(img, 'Previous Track', (10, 70), 
                                      cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
                        else:
                            keyboard.press('right')  # Next track/forward
                            cv2.putText(img, 'Next Track', (10, 70), 
                                      cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
                    
                    elif finger_count == 5:  # Play/Pause
                        keyboard.press('space')
                        cv2.putText(img, 'Play/Pause', (10, 70), 
                                  cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
                    
                    # Display current mode
                    mode_text = {
                        1: "Mouse Control Mode",
                        2: "Volume Control Mode",
                        3: "Brightness Control Mode",
                        4: "Media Navigation Mode",
                        5: "Play/Pause Mode"
                    }.get(finger_count, "")
                    
                    if mode_text:
                        cv2.putText(img, mode_text, (400, 70), 
                                  cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
