import cv2
from filters import apply_filter
from Gesture import detect

cap = cv2.VideoCapture(0)
current_filter = "original"   

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not captured")
            break

        gesture = detect(frame)
        print("Detected gesture:", gesture)

        if gesture == "1_fingers":
            current_filter = "original"
        elif gesture == "2_fingers":
            current_filter = "gray"
        elif gesture == "3_fingers":
            current_filter = "blur"
        elif gesture == "4_fingers":
            current_filter = "edges"
        elif gesture == "5_fingers":
            current_filter = "sepia"

        filtered = apply_filter(frame, current_filter)

        
        if filtered is None:
            print("Filter returned None, check filters.py")
            filtered = frame
        elif len(filtered.shape) == 2:
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

        cv2.putText(filtered,
                    f"Current Filter: {current_filter}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Gesture Controlled Filters", filtered)

        if cv2.waitKey(20) & 0xFF == 27: 
            break

    except Exception as e:
        print("ERROR in loop:", e)
        break

cap.release()
cv2.destroyAllWindows()
