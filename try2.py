import cv2
import numpy as np
import imutils
import crib_child_detection as cd

#Reading From File or 0 to view Live
cap= cv2.VideoCapture('5.mp4')

firstFrame = None

#Loading the Classifies
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')
upper_body = cv2.CascadeClassifier('haarcascade/haarcascade_upperbody.xml')

#take first frame and detect crib until 95% and give a sound until detection in complete
#return the box coordinates

cribDetected = True

ret, frame = cap.read()
cv2.imwrite('test_images/image15.jpg',frame)


while True:
    ret, frame = cap.read()

    frame = imutils.resize(frame,width=480)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=21)
    edges = cv2.Canny(frame,100,100)

    while firstFrame is None:
        #Detect Crib
        #while cribDetected: 
        #    crib_detector(gray)
        continue

    #Detect Faces
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray1)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.rectangle(frame,(cx,cy),(200,100),(0,255,0),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        if( y<80 ):
            safe = False
            print ("Child in Danger Area")
        else:
            safe = True
            print ("Child in Safe Area")

        eyes = eye_cascade.detectMultiScale(gray1)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),1)

        upperbody = upper_body.detectMultiScale(gray)
        for (ub_x,ub_y,ub_w,ub_h) in upperbody:
            cv2.rectangle(frame,(ub_x,ub_y),(ub_x+ub_w,ub_y+ub_h),(255,0,0),2)
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

    
    # Hardcoded CRIB Range
    cv2.rectangle(frame,(60,80),(450,100),(0,0,255),3)
    cv2.rectangle(frame,(60,100),(450,300),(0,255,0),0)
    cv2.imshow('Output',frame)

    if cv2.waitKey(5) == 27:
        break

cap.release()
cv2.destroyAllWindows()

def crib_detector(image):
    (cribDetected, coordinatedic_dic_param) = cd.crib_detector(image)
    if (cribDetected):
        #play Sound CRIB Detected 
        return coordinatedic_dic_param

