import cv2 
import Detect_Crib as dc

import numpy as np

from utils import visualization_utils as vis_util
import utilsSujee as us

def _main():
    cap = cv2.VideoCapture('5.mp4')
    firstframe = None
    
    while True:
        ret,frame = cap.read()
        cv2.imshow('sd',frame)
        #(x,y)=frame.size
        #print (y)
        #print (x)
        #if firstframe is None:
        #    (ymin,xmin, ymax, xmax) = dc.run_detection(frame)
        #    firstframe = frame
        #    cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,0),2)
        #    cv2.imshow('fdg',frame)

        cv2.imshow('Image',frame)
        if cv2.waitKey(5) == 27:
            break
_main()