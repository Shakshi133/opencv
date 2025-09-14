import cv2
import mediapipe as mp

mp_hand=mp.solutions.hands
hands=mp_hand.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.7)
mp_drawing=mp.solutions.drawing_utils

def count_finger(hand_landmark):
    fingers=[]
    landmarks=hand_landmark.landmark

    fingers.append(landmarks[4].x<landmarks[3].x)

    for tip in[8,12,16,20]:
        fingers.append(landmarks[tip].y<landmarks[tip-2].y)

    return fingers.count(True)


cap=cv2.VideoCapture(0)

while cap.isOpened():
    succes,frame=cap.read()
    if not succes:
        break

    frame=cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result=hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hand.HAND_CONNECTIONS)

            finger_count=count_finger(hand_landmarks)
            cv2.putText(frame,f"Fingers:{finger_count}",(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)

    cv2.imshow("Finger Counter",frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

