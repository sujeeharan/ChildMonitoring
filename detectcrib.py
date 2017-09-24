import cv2
import numpy as np


cap=cv2.VideoCapture('5.mp4')

while True:
    _,fram=cap.read()
  
    imgray = cv2.cvtColor(fram,cv2.COLOR_BGR2GRAY)

    fram=cv2.resize(fram,(1080,800))

    lower_red=np.array([127,127,127])
    upper_red=np.array([255,255,255])

    kernal=np.ones((5,5),np.uint8)
    kernal2=np.ones((2,2),np.uint8)
    mask=cv2.inRange(fram,lower_red,upper_red)
    opening=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal)


    ret,thresh = cv2.threshold(imgray,127,255,0)
    image, contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.selectROI('Select Region',fram,True,True)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]


    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(fram,(x,y),(x+w,y+h),(0,255,0),2)

    fram=cv2.resize(fram,(1020,800))
    cv2.imshow('image',fram)
   

    k=cv2.waitKey(24) & 0xff 
    if k==27:
        break


#cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()