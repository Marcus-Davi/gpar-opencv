import cv2
import numpy as np
import time 
from matplotlib import pyplot as plt
import scipy

# Read image. 
img = cv2.imread('bola.jpg', cv2.IMREAD_COLOR)

#Redimensiona imagem
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img,dim)

#Converte imagem de RGB para HSV
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Converte imagem de RGB para P&B
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Teste de cor imagem
for i in range(0,255,1):
    lower_blue = np.array([i,100,100])
    upper_blue = np.array([i+20,255,255])
    print i

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    cv2.imshow('teste',mask)
    cv2.waitKey(220)
'''
for i in range(0,200,10):
    ret, thresh = cv2.threshold(gray,30,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #cv2.imshow("Imagem inicial",img)
    cv2.imshow("Imagem Gray",thresh)
    print(i)
    cv2.waitKey(200)
'''

cv2.destroyAllWindows()
