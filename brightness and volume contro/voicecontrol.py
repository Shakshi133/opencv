import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#system volume interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

#Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


cap = cv2.VideoCapture(0)  # Try 1 or 2 if 0 doesn't work

#Check if camera opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

#Timer setup
start_time = time.time()
VOLUME_CONTROL_DURATION = 60  # seconds
volume_enabled = True

while True:
    success, img = cap.read()

    if not success or img is None:
        print("Warning: Failed to grab frame from webcam.")
        continue

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    current_time = time.time()
    if current_time - start_time > VOLUME_CONTROL_DURATION:
        volume_enabled = False

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2]   # Thumb tip
                x2, y2 = lmList[8][1], lmList[8][2]   # Index tip

                # Draw circles and line
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Distance between fingers
                length = math.hypot(x2 - x1, y2 - y1)

                if volume_enabled:
                    # Map length to volume [0.0, 1.0]
                    vol_scalar = np.interp(length, [50, 250], [0.0, 1.0])
                    volume.SetMasterVolumeLevelScalar(vol_scalar, None)

                    # Volume bar and percentage
                    vol_bar = np.interp(length, [50, 250], [400, 150])
                    vol_percent = int(vol_scalar * 100)

                    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, f'{vol_percent} %', (40, 430), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2)
                else:
                    # Timer expired
                    cv2.putText(img, "Volume control expired", (200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Display window
    cv2.imshow("Hand Volume Control", img)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


