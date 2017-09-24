import cv2 
import imutils
import numpy as np

video=cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    #cv2.imshow("Fam",frame)
    frame = imutils.resize(frame,width=480)
    frame =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("check", frame)
    print('sad')
    if cv2.waitKey(5) == 27:
        break

    
    