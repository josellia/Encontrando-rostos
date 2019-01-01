# -*- coding: UTF-8 -*-
"""Detector de faces"""

import cv2

def detectaFaces():
    video_capture = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        # Capturar quadro a quadro
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Desenhar o retângulo em torno do rosto
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Exibe o quadro resultante
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'): #utlize a tecla s minuscula do teclado para fechar a janela
            break

    # Quando tudo estiver pronto, libere a captura
    video_capture.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    detectaFaces()