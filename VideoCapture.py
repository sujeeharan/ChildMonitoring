import cv2 

videoCapture = cv2.VideoCapture('1.mp4')
fps= videoCapture.get(cv2.CAP_PROP_FPS)
size = ((int)(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
		 (int)(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter("output.avi",cv2.CAP_PROP_FOURCC('I','4','2','0'), fps)

success, frame = videoCapture.read() 
while success: # Loop until there are no more frames.    
	videoWriter.write(frame)    
	success, frame = videoCapture.read()

