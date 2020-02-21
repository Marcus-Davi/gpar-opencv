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
txt_ft = 'Filtro'

#############################################

# CALIBRA INFERIOR
paletainf = np.zeros((200, 200, 3), np.uint8)
cv2.namedWindow(txt_inf)

# create trackbars for color change
cv2.createTrackbar('H', txt_inf, 100, 180, nothing)
cv2.createTrackbar('S', txt_inf, 90, 255, nothing)
cv2.createTrackbar('V', txt_inf, 70, 255, nothing)

#############################################

# CALIBRA SUPERIOR
paletasup = np.zeros((200, 200, 3), np.uint8)
cv2.namedWindow(txt_sup)

# create trackbars for color change
cv2.createTrackbar('H', txt_sup, 130, 180, nothing)
cv2.createTrackbar('S', txt_sup, 255, 255, nothing)
cv2.createTrackbar('V', txt_sup, 255, 255, nothing)

#############################################

# CAIXA FILTROS

cx_filtro = np.zeros((1, 300, 3), np.uint8)
cv2.namedWindow(txt_ft)
cv2.createTrackbar('Denoise', txt_ft, 0, 20, nothing)
# cv2.createTrackbar('Denoise_qtde',txt_ft,0,20,nothing)
cv2.createTrackbar('Erosao', txt_ft, 0, 10, nothing)
# cv2.createTrackbar('Erosao_qtde',txt_ft,0,10,nothing)
cv2.createTrackbar('Dilatacao', txt_ft, 0, 10, nothing)
# cv2.createTrackbar('Dilatacao_qtde',txt_ft,0,10,nothing)
cv2.createTrackbar('Erosao_Dilatacao', txt_ft, 0, 10, nothing)
# cv2.createTrackbar('Erosao_Dilatacao_qtde',txt_ft,0,10,nothing)
cv2.createTrackbar('Blur', txt_ft, 0, 10, nothing)
# cv2.createTrackbar('Blur_qtde',txt_ft,0,10,nothing)
forma_detect = '0 : Circ \n1 : Quad'
cv2.createTrackbar(forma_detect, txt_ft, 1, 1, nothing)
cv2.createTrackbar('Par1', txt_ft, 50, 100, nothing)
cv2.createTrackbar('Par2', txt_ft, 30, 100, nothing)
cv2.createTrackbar('Raio_min', txt_ft, 15, 100, nothing)
cv2.createTrackbar('Raio_max', txt_ft, 80, 300, nothing)
cv2.createTrackbar('Area', txt_ft, 2000, 10000, nothing)

###################################

# Read image.
cap = cv2.VideoCapture(0)


while(1):

#####################################################################
############ CAPTURA DE IMAGEM ######################################
#####################################################################

    # Captura imagem
    _, img_orig = cap.read()

    # Inverte imagem para deixar "correta" a exibicao
    img_orig = cv2.flip(img_orig, 1)

    # Redimensiona imagem
    scale_percent = 60  # percent of original size
    width = int(img_orig.shape[1] * scale_percent / 100)
    height = int(img_orig.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_orig = cv2.resize(img_orig, dim)

#####################################################################
########### FILTROS GUIDE ###########################################
#####################################################################

    #######################################

    #### CALIBRACAO INFERIOR #####
    cv2.imshow(txt_inf, paletainf)
    hinf = cv2.getTrackbarPos('H', txt_inf)
    sinf = cv2.getTrackbarPos('S', txt_inf)
    vinf = cv2.getTrackbarPos('V', txt_inf)
    paletainf[:] = [hinf, sinf, vinf]
    pinf = paletainf[1][1][:]
    paletainf = cv2.cvtColor(paletainf, cv2.COLOR_HSV2BGR)

    #######################################

    #### CALIBRACAO SUPERIOR #####
    cv2.imshow(txt_sup, paletasup)
    hsup = cv2.getTrackbarPos('H', txt_sup)
    ssup = cv2.getTrackbarPos('S', txt_sup)
    vsup = cv2.getTrackbarPos('V', txt_sup)
    paletasup[:] = [hsup, ssup, vsup]
    psup = paletasup[1][1][:]
    paletasup = cv2.cvtColor(paletasup, cv2.COLOR_HSV2BGR)

    #######################################

    #### CAIXA DE FILTROS #####
    # Pegando valores da caixa
    cv2.imshow(txt_ft, cx_filtro)
    erose_sw = cv2.getTrackbarPos('Erosao', txt_ft)
    # erose_qt = cv2.getTrackbarPos('Erosao_qtde',txt_ft)
    dilation_sw = cv2.getTrackbarPos('Dilatacao', txt_ft)
    # dilation_qt = cv2.getTrackbarPos('Dilatacao_qtde',txt_ft)
    eros_dil_sw = cv2.getTrackbarPos('Erosao_Dilatacao', txt_ft)
    denoise_sw = cv2.getTrackbarPos('Denoise', txt_ft)
    # denoise_qt = cv2.getTrackbarPos('Denoise_qtde',txt_ft)
    blur_sw = cv2.getTrackbarPos('Blur', txt_ft)
    # blur_qt = cv2.getTrackbarPos('Blur_qtde',txt_ft)
    par1_bar = cv2.getTrackbarPos('Par1', txt_ft)
    if par1_bar <= 0: par1_bar = 1
    par2_bar = cv2.getTrackbarPos('Par2', txt_ft)
    if par2_bar <= 0: par2_bar = 1
    raio_min = cv2.getTrackbarPos('Raio_min', txt_ft)
    raio_max = cv2.getTrackbarPos('Raio_max', txt_ft)
    formato_det = cv2.getTrackbarPos(forma_detect, txt_ft)
    area_rect = cv2.getTrackbarPos('Area', txt_ft)

#####################################################################
####### IMPLEMENTA FILTROS DA GUIDE #################################
#####################################################################
    result = img_orig
    #######################################

    # Removendo ruidos
    if denoise_sw == 0:
        result = img_orig
        cv2.destroyWindow('Denoising')
    else:
        img_denoised = cv2.fastNlMeansDenoisingColored(
            img_orig, None, 10, 10, 7, denoise_sw)
        result = img_denoised
        cv2.imshow('Denoising', img_denoised)

    #######################################

    # Converte imagem de RGB para HSV
    img_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    # Define cores detectaveis
    lower_blue = np.array([pinf])
    upper_blue = np.array([psup])

    # Threshold the HSV image to get only blue colors
    result = cv2.inRange(img_hsv, lower_blue, upper_blue)

    #######################################

    # Filtragem de imagem - Reduz ruidos
    if erose_sw == 0:
        result = result
        cv2.destroyWindow('erosion')
    else:
        nf = erose_sw
        kernel_er = np.ones((nf, nf), np.uint8)
        result = cv2.erode(result, kernel_er, iterations=3)
        erosed = result
        cv2.imshow('erosion', erosed)

    #######################################

    # Filtragem de imagem - Preenche vazios
    if dilation_sw == 0:
        cv2.destroyWindow('dilation')
    else:
        nf = dilation_sw
        kernel_dil = np.ones((nf, nf), np.uint8)
        result = cv2.dilate(result, kernel_dil, iterations=3)
        dilation = result
        cv2.imshow('dilation', dilation)

    #######################################

    # Filtragem de imagem - Erosion seguido de dilation

    if eros_dil_sw == 0:
        cv2.destroyWindow('Erosion_plus_Dilation')
    elif erose_sw == 0 and dilation_sw == 0:
        nf = eros_dil_sw
        kernel_opn = np.ones((nf, nf), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_opn)
        opening = result
        cv2.imshow('Erosion_plus_Dilation', opening)

    #######################################

    # BORRA A IMAGEM
    # Adiciona filtro, borrando imagem
    if blur_sw == 0:
        cv2.destroyWindow('Img_Borrada')
    else:
        result = cv2.GaussianBlur(result, (5, 5), 0)
        opening = result
        cv2.imshow('Img_Borrada', opening)

    #########################################################
    ########## DETECCAO DE CIRCULOS #########################
    #########################################################

    if formato_det == 0:

        detected_circles = cv2.HoughCircles(result,
                                            cv2.HOUGH_GRADIENT, 1, 20, param1=par1_bar,
                                            param2=par2_bar, minRadius=raio_min, maxRadius=raio_max)

        # Draw circles that are detected.
        if detected_circles is not None:
  
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            #print(detected_circles)

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                cv2.circle(img_orig, (a, b), r, (0, 255, 0), 2)

                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(img_orig, (a, b), 1, (255, 0, 0), 3)

        
    elif formato_det == 1:

    #########################################################
    ########## DETECCAO DE RETANGULOS #########################
    #########################################################

        (_, contours, _) = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if(area > area_rect):
                x, y, w, h = cv2.boundingRect(contour)  
                frame_rect = cv2.rectangle(img_orig, (x, y), (x+w, y+h), (0, 0, 255), 10)
                pos = [x,y,w,h]
                print(pos)

    cv2.imshow("Imagem com filtros", result)
    cv2.imshow("Imagem original", img_orig)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
