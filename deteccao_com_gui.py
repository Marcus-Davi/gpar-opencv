import cv2
import numpy as np
import time

def nothing(x):
    pass

#############################################
### GUIDE PARA AJUDAR NA DETECCAO DE CORES###
#############################################

# Create a black image, a window
txt_inf = 'CALIBRA_INFERIOR'
txt_sup = 'CALIBRA_SUPERIOR'

# CALIBRA INFERIOR
paletainf = np.zeros((200,200,3), np.uint8)
cv2.namedWindow(txt_inf)

# create trackbars for color change
cv2.createTrackbar('H',txt_inf,0,180,nothing)
cv2.createTrackbar('S',txt_inf,0,255,nothing)
cv2.createTrackbar('V',txt_inf,0,255,nothing)

# CALIBRA SUPERIOR
paletasup = np.zeros((200,200,3), np.uint8)
cv2.namedWindow(txt_sup)

# create trackbars for color change
cv2.createTrackbar('H',txt_sup,0,180,nothing)
cv2.createTrackbar('S',txt_sup,0,255,nothing)
cv2.createTrackbar('V',txt_sup,0,255,nothing)


# Read image.
#img = cv2.imread('bola.jpg', cv2.IMREAD_COLOR)
cap = cv2.VideoCapture(0)


while(1):

    #### CALIBRACAO INFERIOR #####
    cv2.imshow(txt_inf,paletainf)
    hinf = cv2.getTrackbarPos('H',txt_inf)
    sinf = cv2.getTrackbarPos('S',txt_inf)
    vinf = cv2.getTrackbarPos('V',txt_inf)
    paletainf[:] = [hinf,sinf,vinf]
    pinf = paletainf[1][1][:]
    paletainf = cv2.cvtColor(paletainf, cv2.COLOR_HSV2BGR)

    #### CALIBRACAO SUPERIOR #####
    cv2.imshow(txt_sup,paletasup)
    hsup = cv2.getTrackbarPos('H',txt_sup)
    ssup = cv2.getTrackbarPos('S',txt_sup)
    vsup = cv2.getTrackbarPos('V',txt_sup)
    paletasup[:] = [hsup,ssup,vsup]
    psup = paletasup[1][1][:]
    paletasup = cv2.cvtColor(paletasup, cv2.COLOR_HSV2BGR)
   
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
     
    # Converte imagem de RGB para HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define cores detectaveis
    lower_blue = np.array([pinf])
    upper_blue = np.array([psup])
    #print pinf1
    #print pinf_hsv_1r1c
    
    # Threshold the HSV image to get only blue colors
    pb = cv2.inRange(img_hsv, lower_blue, upper_blue)
    #cv2.imshow('Imagem P&B', pb)
    # cv2.waitKey(0)

    # Filtragem de imagem - Reduz ruidos    
    nf = 1
    kernel_er = np.ones((nf, nf), np.uint8)
    erosion = cv2.erode(pb, kernel_er, iterations=3)
    cv2.imshow('erosion', erosion)
    # cv2.waitKey(0)

    # Filtragem de imagem - Preenche vazios
    nf = 2
    kernel_dil = np.ones((nf, nf), np.uint8)
    dilation = cv2.dilate(erosion, kernel_dil, iterations=1)
    cv2.imshow('dilation', dilation)
    # cv2.waitKey(0)

    # Filtragem de imagem - Erosion seguido de dilation
    nf = 2
    kernel_opn = np.ones((nf, nf), np.uint8)
    opening = cv2.morphologyEx(pb, cv2.MORPH_OPEN, kernel_opn)
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
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        
    cv2.imshow("Imagem filtrada", opening)
    cv2.imshow("Deteccao", img)
    cv2.imshow("Imagem sem filtro", pb)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
        
cap.release()

cv2.destroyAllWindows()
