'''
Transforma as imagens originais do dataset em imagens com recortes das mãos
'''

import numpy as np
import pandas as pd
import os
import cv2
import gc
from cvzone.HandTrackingModule import HandDetector
from handDetection import detectHands 

# Utilizar também pastas para as imagens de treino e de teste
outputFolder = "ImagensProcessadasValidation"

df = pd.read_csv("projeto libras.v21i.tensorflow/valid/_annotations.csv")

i = 0

# itera sobre as linhas do dataframe
for index, row in df.iterrows():
    fileName = row["filename"] # Nome do arquivo da imagem
    targetClass = row["class"]

    inputImagePath = os.path.join("projeto libras.v21i.tensorflow/valid", fileName)

    try:
        img = cv2.imread(inputImagePath)

        detector = HandDetector(staticMode=True, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.2)

        # Coordenadas de recorte da/das mão/mãos
        coordinates = detectHands(img, detector)

        if coordinates is None:
            print("Mão não encontrada!")
        else:
            print(inputImagePath)
            xMin, xMax, yMin, yMax = coordinates
            
            # Ajusta os parâmetros
            xMin -= 30
            xMax += 30
            yMin -= 30
            yMax += 30

            cropImg = img[yMin:yMax, xMin:xMax]

            # Localiza e cria (se necessário) a pasta
            classFolder = os.path.join(outputFolder, targetClass)
            os.makedirs(classFolder, exist_ok=True)

            outputImagePath = os.path.join(classFolder, f"{targetClass}_{i}.jpg")

            # Salva a imagem
            cv2.imwrite(outputImagePath, cropImg)
            i += 1

    except Exception as e:
        print(f"Erro ao processar {fileName}: {e}")




