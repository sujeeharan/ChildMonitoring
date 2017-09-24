import cv2
import numpy as np
import imutils

#Reading From File or 0 to view Live
cap= cv2.VideoCapture('5.mp4')

#reading from a image file Testing 
#frame = cv2.imread('6.jpg')
#firstFrame = cv2.imread('firstframe.jpg',3)
firstFrame = None
#firstFrame = imutils.resize(firstFrame,width=500)
#print (firstFrame.shape)
#Loading the Classifies
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')
upper_body = cv2.CascadeClassifier('haarcascade/haarcascade_upperbody.xml')


while True:
    ret, frame = cap.read()

    frame = imutils.resize(frame,width=480)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=21)
    edges = cv2.Canny(frame,100,100)

    if firstFrame is None:
        firstFrame = gray
        continue

    ##TEsting Corners
    ##gray = np.float32(gray1)

    #corners = cv2.goodFeaturesToTrack(gray,100,0.01,100)
    #corners = np.int0(corners)
    #xmin,ymin = corners[0].ravel()

    ##Tracking the Child
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (21, 21), 0)

    #Find the difference between Current frame and First frame
    
  #  frameDelta = cv2.absdiff(firstFrame, gray)
  #  thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
  #  thresh = cv2.dilate(gray, None, iterations=2)
  #  (cnts, _s,_1) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,
		#cv2.CHAIN_APPROX_SIMPLE)

  #  for c in cnts:
		## if the contour is too small, ignore it
  #      #if cv2.contourArea(c) < args["min_area"]:
  #      #    continue
 
		## compute the bounding box for the contour, draw it on the frame,
		## and update the text
  #      (childx, childy, childw, childh) = cv2.boundingRect(c)
  #      cv2.rectangle(gray, (childx, childy), (childx + childw, childy + childh), (0, 0, 255), 2)


    # Hardcoded CRIB Range
    cv2.rectangle(frame,(60,40),(450,80),(0,0,255),3)
    cv2.rectangle(frame,(60,80),(450,300),(0,255,0),0)

    #Detect Faces
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray1)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.rectangle(frame,(cx,cy),(200,100),(0,255,0),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        if(x>60 & y>40):
            print ("Child in Danger Area")

        eyes = eye_cascade.detectMultiScale(gray1)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),1)


    cv2.imshow('FaceDetection',gray1)
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    thresh = cv2.dilate(gray, None, iterations=2)
    (cnts, _s,_1) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(thresh,2,3,0.1)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

#result is dilated for marking the corners, not important
    #dst = cv2.dilate(dst,None)
    #res = np.hstack((centroids,corners))
    #res = np.int0(res)
    #frame[res[:,1],res[:,0]]=[0,0,255]
    #frame[res[:,3],res[:,2]] = [0,255,0]

    cv2.imshow('subpixel5',frame)

# Threshold for an optimal value, it may vary depending on the image.
    #frame1[dst>0.01*dst.max()]=[0,0,255]
    #cv2.imshow('test',frame1)

    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(frame,(x,y),3,255,-1)

    cv2.imshow('Test1',frame)
    #cv2.imshow('Sobel',edges)

    if cv2.waitKey(5) == 27:
        break

cap.release()
cv2.destroyAllWindows()