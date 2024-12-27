'''
Determina as coordenadas do bounding box que compreende as mãos detectadas
'''

from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

def detectHands(image, detector):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hands, img = detector.findHands(image_rgb, draw=False, flipType=True)

    # Verifica se alguma mão foi identificada
    if hands:
        # Para a primeira mão
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # Lista dos 21 landmarks
        bbox1 = hand1["bbox"]  # bounding box da mão

        # x e y: coordenadas horizontais e verticais iniciais
        # w e h: comprimento e altura
        x1, y1, w1, h1 = bbox1

        # Checa pela segunda mão
        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]

            x2, y2, w2, h2 = bbox2
            
            # Retorna um bounding box que comporta ambas as mãos
            xMin = min(x1, x2)
            xMax = max(x1 + w1, x2 + w2)
            yMin = min(y1, y2)
            yMax = max(y1 + h1, y2 + h2)

            return xMin, xMax, yMin, yMax

        return x1, x1 + w1, y1, y1 + h1
    
    return None