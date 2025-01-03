import pickle
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp

with open("modelo_sign_language.pkl", 'rb') as file:
    model = pickle.load(file)
    
mp_hand = mp.solutions.hands
Hand = mp_hand.Hands()
mp_draw = mp.solutions.drawing_utils
connections_style = mp_draw.DrawingSpec(color=(0, 255, 0))

video = cv2.VideoCapture(0)

while True:
    ok, frame = video.read()
    if not ok:
        break
    
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = Hand.process(frameRGB)
    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handlm, mp_hand.HAND_CONNECTIONS, connection_drawing_spec=connections_style)
            hand = handlm.landmark  
            hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand]).flatten())
       
            X = pd.DataFrame([hand_row])
            hand_class = model.predict(X)[0]
            cv2.putText(frame, str(hand_class), (0,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()