import numpy as np
import cv2

cap = cv2.VideoCapture(0)


while(1):

    # Take each frame
    _, frame = cap.read()

    # marcus incluido
    blue = np.uint8([[[0,255,0 ]]])
    hsv_blue = cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
#    print hsv_blue

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # define range of blue color in HSV
    lower_blue = np.array([110,80,80])
    upper_blue = np.array([133,255,225])


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    (_,contours,_) = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if(area > 800):
		x,y,w,h = cv2.boundingRect(contour)
		frame_rect = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),10)

                pos = [x,y, w, h]
                print(pos)
        	cv2.imshow("tracking", frame_rect)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    

cv2.destroyAllWindows()
