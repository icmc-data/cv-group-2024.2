import cv2
from ultralytics import YOLO

# Carrega o modelo
yolo = YOLO("best-YOLOn.pt")

# Captura o video
videoCap = cv2.VideoCapture(0)

# Loop
while True:
    # Le a captura
    ret, frame = videoCap.read()
    if not ret:
        continue
    # Passa pelo modelo
    results = yolo.track(frame, stream=True)

    # Se tiver resultados, aplica eles no frame
    for result in results:
        # Pega os nomes
        classes_names = result.names

        # Constroi as caixas
        for box in result.boxes:
            if box.conf[0] > 0.5:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])

                class_name = classes_names[cls]

                # Adiciona os retangulo em volta do sinal de libras
                cv2.rectangle(frame, (x1, y1), (x2, y2), 2)

                # Adiciona o texto acima do retangulo
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, 2)
                
    # Mostra a imagem
    cv2.imshow('frame', frame)

    # Quebra o loop ao apertar b
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Libera a webcam e fecha janelas
videoCap.release()
cv2.destroyAllWindows()
