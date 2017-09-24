import numpy as np
import cv2

#frame = cv2.imread('1.jpg')

cap = cv2.VideoCapture('5.mp4')

lower = np.array([0, 0, 0])
upper = np.array([15, 15, 15])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()


while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    #edged
    edged = cv2.Canny(blurred, 50, 150)
    

    #THRESHOLD
    ret,thresh = cv2.threshold(gray,100,255,0)
    
    #CORNERS
    corners = cv2.goodFeaturesToTrack(blurred, 10,0.01,100)
    corners = np.int0(corners)
    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(thresh,(x,y),10,255,-1)

    #Background substraction
    fgmask = fgbg.apply(thresh)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    edged = cv2.Canny(thresh, 50, 150)
    

    dst = cv2.addWeighted(fgmask,0.3,thresh,0.6,0.1)

    #Thresh and back substr
    cv2.rectangle(frame,(90,140),(100+500,100+350),(255,0,0),4)
    pts = [(90,600),(100,450)] 
    boundrect = cv2.boundingRect(pts)
    cv2.selectROI('ROI',frame,boundrect,None)
    cv2.imshow('Original',frame)
    #corners in thresh
    cv2.imshow('frameasd',thresh)
    #fgmask edged
    cv2.imshow('frame',edged)

    k = cv2.waitKey(1) & 0xff
    if k == 27 :    
        break



cap.release()
cv2.destroyAllWindows()