import cv2
import numpy as np

cap = cv2.VideoCapture('3.mp4')

while True:
    _,frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 100,0.01,10)
    corners = np.int0(corners)

    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(frame,(x,y),3,255,-1)

    cv2.imshow('Corners',frame)

    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
