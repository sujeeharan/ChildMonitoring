import numpy as np
import cv2

img = cv2.imread('one.jpg',cv2.IMREAD_COLOR)

cv2.line(img,(10,20),(150,150),(255,152,0),15)  
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
