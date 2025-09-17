import time
import math
import numpy as np
import pyautogui
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


screen_w, screen_h = pyautogui.size()

prev_x, prev_y = 0.0, 0.0
smoothening = 5.0          
alpha = 1.0 / smoothening


frames_required = 3        
click_cooldown = 0.5      
click_counters = {'left': 0, 'right': 0, 'double': 0}
last_click_time = {'left': 0.0, 'right': 0.0, 'double': 0.0}


def to_pixel(lm, img_w, img_h):
    return int(lm.x * img_w), int(lm.y * img_h)

def distance_px(lm1, lm2, img_w, img_h):
    x1, y1 = to_pixel(lm1, img_w, img_h)
    x2, y2 = to_pixel(lm2, img_w, img_h)
    return math.hypot(x2 - x1, y2 - y1)


prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not available")
            break

        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        status_text = ""
        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            index_lm = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_lm = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_lm = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_lm = hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            wrist_lm = hand.landmark[mp_hands.HandLandmark.WRIST]
            middle_mcp_lm = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            ix, iy = to_pixel(index_lm, img_w, img_h)
            screen_x = np.interp(ix, [0, img_w], [0, screen_w])
            screen_y = np.interp(iy, [0, img_h], [0, screen_h])

            if prev_x == 0.0 and prev_y == 0.0:
                cur_x, cur_y = screen_x, screen_y
            else:
                cur_x = prev_x * (1 - alpha) + screen_x * alpha
                cur_y = prev_y * (1 - alpha) + screen_y * alpha


            try:
                pyautogui.moveTo(int(cur_x), int(cur_y))
            except Exception:
        
                pass

            prev_x, prev_y = cur_x, cur_y

            hand_size_px = distance_px(wrist_lm, middle_mcp_lm, img_w, img_h)
            if hand_size_px < 1:
                hand_size_px = 100
            thresh = max(25, int(hand_size_px * 0.35))  # tweak multiplier if needed

 
            d_index_thumb = distance_px(index_lm, thumb_lm, img_w, img_h)
            d_middle_thumb = distance_px(middle_lm, thumb_lm, img_w, img_h)
            d_ring_thumb = distance_px(ring_lm, thumb_lm, img_w, img_h)

            now = time.time()

            if d_index_thumb < thresh:
                click_counters['left'] += 1
                if click_counters['left'] >= frames_required and (now - last_click_time['left'] > click_cooldown):
                    pyautogui.click()
                    last_click_time['left'] = now
                    status_text = "Left click"
                    click_counters['left'] = 0
            else:
                click_counters['left'] = 0

       
            if d_middle_thumb < thresh:
                click_counters['right'] += 1
                if click_counters['right'] >= frames_required and (now - last_click_time['right'] > click_cooldown):
                    pyautogui.rightClick()
                    last_click_time['right'] = now
                    status_text = "Right click"
                    click_counters['right'] = 0
            else:
                click_counters['right'] = 0

            if d_ring_thumb < thresh:
                click_counters['double'] += 1
                if click_counters['double'] >= frames_required and (now - last_click_time['double'] > click_cooldown):
                    pyautogui.doubleClick()
                    last_click_time['double'] = now
                    status_text = "Double click"
                    click_counters['double'] = 0
            else:
                click_counters['double'] = 0

            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)
            if status_text:
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Virtual Mouse (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
