import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import scipy

# Read image.
#img = cv2.imread('bola.jpg', cv2.IMREAD_COLOR)
cap = cv2.VideoCapture(0)
'''
_, img = cap.read()
cv2.imshow('teste',img)

#cv2.waitKey(0)

k = cv2.waitKey(1) & 0xFF
if k == 27:
    cv2.destroyAllWindows()

'''
while(1):
    # Take each frame
    _, img = cap.read()

    # Inverte imagem para deixar "correta" a exibicao
    img = cv2.flip(img,1)

    # Redimensiona imagem
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim)

    # Borra imagem para reduzir ruidos
    #borra_img = cv2.GaussianBlur(img,(5,5),0)

    # Converte imagem de RGB para HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Converte imagem de RGB para P&B
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecta cor vermelha
    lower_blue = np.array([0, 100, 100])
    upper_blue = np.array([8, 255, 255])

    # Threshold the HSV image to get only blue colors
    pb = cv2.inRange(img_hsv, lower_blue, upper_blue)
    #cv2.imshow('Imagem P&B', pb)
    # cv2.waitKey(0)

    # Filtragem de imagem - Reduz ruidos
    nf = 2
    kernel = np.ones((nf, nf), np.uint8)
    erosion = cv2.erode(pb, kernel, iterations=3)
    #cv2.imshow('erosion', erosion)
    # cv2.waitKey(0)

    # Filtragem de imagem - Preenche vazios
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    #cv2.imshow('dilation', dilation)
    # cv2.waitKey(0)

    # Filtragem de imagem - Erosion seguido de dilation
    opening = cv2.morphologyEx(pb, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('opening', opening)
    # cv2.waitKey(0)

    # Adiciona filtro, borrando imagem
    img_ftd = cv2.GaussianBlur(opening, (5, 5), 0)

    detected_circles = cv2.HoughCircles(img_ftd,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=1, maxRadius=80)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        print(detected_circles)
 
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(opening, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(opening, (a, b), 1, (0, 0, 255), 3)
        
    cv2.imshow("Detected Circle", opening)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
        
cap.release()

cv2.destroyAllWindows()
