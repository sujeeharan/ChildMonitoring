import numpy as np
import cv2

img1 = cv2.imread('1.jpg')

img2 = cv2.imread('2.jpg')

#img1 = cv2.resize(img1,(100,100))
#img2 = cv2.resize(img2,(100,100))
#add= img1+img2 

rows, cols,channels = img1.shape
roi = img1[0:rows,0:cols]

img2gray= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray,220,255,cv2.THRESH_BINARY_INV)
 


cv2.imshow('add',img2gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
