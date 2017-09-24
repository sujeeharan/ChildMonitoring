import cv2
cameraCapture = cv2.VideoCapture(0) 
fps = 30 
# an assumption 
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),        
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))) 

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

#videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.CAP_PROP_FOURCC('I','4','2','0'), fps, size)
success, frame = cameraCapture.read() 
numFramesRemaining = 10 * fps - 1 
while success and numFramesRemaining > 0:    
    out.write(frame)    
    success, frame = cameraCapture.read()    
    numFramesRemaining -= 1

#import numpy
#import cv2

#cap = cv2.VideoCapture(0)

#while (True):
#    ret, frame = cap.read()

#    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#    cv2.imshow('frame',gray)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
#cv2.destroyAllWindows()