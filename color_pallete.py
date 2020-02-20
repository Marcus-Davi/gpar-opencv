import cv2
import numpy as np
import time

def nothing(x):
    pass

# Create a black image, a window
paleta = np.zeros((200,200,3), np.uint8)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
#switch = '0 : OFF \n1 : ON'
#cv2.createTrackbar(switch, 'image',0,1,nothing)

while(1):
    cv2.imshow('image',paleta)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    #s = cv2.getTrackbarPos(switch,'image')

    #if s == 0:
    #    img[:] = 0
    #else:
    paleta[:] = [b,g,r]
    pcvt = cv2.cvtColor(paleta,cv2.COLOR_BGR2HSV)
    pl = pcvt[1][1][:]
    print pl
    time.sleep(0.1)

cv2.destroyAllWindows()