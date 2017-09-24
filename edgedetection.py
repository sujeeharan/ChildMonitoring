import cv2
import numpy as np

cap = cv2.VideoCapture('3.mp4')
while True:
    _,frame = cap.read()
    
    laplacian= cv2.Laplacian(frame,cv2.CV_64F)
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=21)
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=21)
    edges = cv2.Canny(frame,100,100)

    cv2.imshow('Original',frame)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('X',sobelx)
    cv2.imshow('Y',sobely)
    cv2.imshow('EDge',edges)

    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break
cv2.destroyAllWindow()
cap.release()