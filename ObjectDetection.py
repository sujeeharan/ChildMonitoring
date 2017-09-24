import cv2
import numpy as np

img = cv2.VideoCapture('5.jpg')

while True:
    img_filt = cv2.medianBlur(img,20)
    img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('dh',img_th)
