import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from playsound import playsound

from utils import label_map_util
from utils import visualization_utils as vis_util


def load_model():
    MODEL_NAME = 'crib_detection_graph'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'cribdetection.pbtxt')

    NUM_CLASSES = 1

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image15 - Copy ({}).jpg'.format(i)) for i in range(2, 6) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

#Coordinates of the Crib
coordinatedic_dic_param={}

def crib_detector(raw_image):
    load_model()
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:

        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

        for (box,score,crib,num_of_crib) in (boxes, scores, classes, num):
            if ((num_of_crib>1)):
                #playSound
                playsound('/sound/tick.wav')
            if (crib_Not_Detected):
                if (score> 90):
                    coordinatedic_dic_param=vis_util.visualize_boxes_and_labels_on_image_array(
                      image_np,
                      np.squeeze(box),
                      np.squeeze(crib).astype(np.int32),
                      np.squeeze(score),
                      category_index,
                      use_normalized_coordinates=True,
                      line_thickness=8)
            else:
                break



        if coordinatedic_dic_param is not None:
              coordinatedic_dic_param=coordinatedic_dic_param
              print(coordinatedic_dic_param['top'])
              cribDetected=true
        else:
            cribDetected= False     
        
    if coordinatedic_dic is not None:
        ch.chiddetectmethod(cap,coordinatedic_dic)
    
    return (cribDetected,coordinatedic_dic_param)
        
        #for image_path in TEST_IMAGE_PATHS:
        #  image = Image.open(image_path)
        #  print(image_path)
        #  # the array based representation of the image will be used later in order to prepare the
        #  # result image with boxes and labels on it.
        #  image_np = load_image_into_numpy_array(image)
        #  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        #  image_np_expanded = np.expand_dims(image_np, axis=0)
        #  # Actual detection.
        #  (boxes, scores, classes, num) = sess.run(
        #      [detection_boxes, detection_scores, detection_classes, num_detections],
        #      feed_dict={image_tensor: image_np_expanded})

        #  #Our Detection
        #  print (boxes)

        #  # Visualization of the results of a detection.
        #  coordinatedic_dic_param=vis_util.visualize_boxes_and_labels_on_image_array(
        #      image_np,
        #      np.squeeze(boxes),
        #      np.squeeze(classes).astype(np.int32),
        #      np.squeeze(scores),
        #      category_index,
        #      use_normalized_coordinates=True,
        #      line_thickness=8)
      
        #  if coordinatedic_dic_param is not None:
        #      coordinatedic_dic_param=coordinatedic_dic_param
        #      print(coordinatedic_dic_param['top'])
        #      break      
        #  #plt.figure(figsize=IMAGE_SIZE)
        #  #plt.imshow(image_np)
        #  #cv2.imshow('Detection',image_np)
