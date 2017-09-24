import cv2
import numpy as np
from matplotlib import pyplot as pypl

img = cv2.imread('one.jpg',0)
edges = cv2.Canny(img,100,200)
pypl.subplot(121),pypl.imshow(img,cmap = 'gray')
pypl.title('Original Image'), pypl.xticks([]), pypl.yticks([])
pypl.subplot(122),pypl.imshow(edges,cmap = 'gray')
pypl.title('Edge Image'), pypl.xticks([]), pypl.yticks([])
pypl.show()

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    edges = cv2.Canny(ret,100,200)
    pypl.subplot(121),pypl.imshow(img,cmap = 'gray')
    pypl.title('Original Image'), pypl.xticks([]), pypl.yticks([])
    pypl.subplot(122),pypl.imshow(edges,cmap = 'gray')
    pypl.title('Edge Image'), pypl.xticks([]), pypl.yticks([])
    pypl.show()

