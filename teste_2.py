import cv2
import numpy as np

# Read image. 
img = cv2.imread('bola.jpg', cv2.IMREAD_COLOR) 


#cv2.waitKey(0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
cv2.imshow("Imagem inicial",img)
cv2.imshow("Laplacian",laplacian)
cv2.waitKey(0)











cv2.destroyAllWindows()