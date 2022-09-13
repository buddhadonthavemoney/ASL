import cv2
import time
import mediapipe as mp
import sys

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
ptime = 0
ctime = 0
prediction = "a"

def get_result():
    time.sleep(1)
    return "detection"

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1080,720), interpolation= cv2.INTER_AREA)
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    h, w, c = img.shape 
    counter = 0
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x + 200
                if x < x_min:
                    x_min = x - 200
                if y > y_max:
                    y_max = y + 200
                if y < y_min:
                    y_min = y - 200
            hand_img = img[y_min:y_max, x_min: x_max]

            if counter%5 == 0:
                try:
                    hand_img = cv2.resize(hand_img, (200,200), interpolation = cv2.INTER_AREA)
                    cv2.imshow("hand", hand_img)
                    # prediction = get_result()
                    cv2.imwrite("hand.jpg", hand_img)
                except Exception as e:
                    print(str(e))
            counter+=1

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cv2.putText(img, f"{prediction}{counter}", (x_min, y_max+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            # mpDraw.draw_landmarks(img, handLms)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
            (255, 0, 255), 3
        )
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


