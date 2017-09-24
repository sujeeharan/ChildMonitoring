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
import utilsSujee as crib


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



def crib_detector(raw_image):
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
    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)

    #Coordinates of the Crib
    coordinatedic_dic_param={}
    coordinatedic_dic={}
    image_np= load_image_into_numpy_array(raw_image)
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.Session(graph=detection_graph) as sess:
            
            print('running detection')
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
            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})

            ymin,xmin,ymax,xmax,status = crib.get_coordinates_from_tensor(image_np_expanded,
                                                                          np.squeeze(boxes),
                                                                          np.squeeze(classes).astype(np.int32),
                                                                          np.squeeze(scores),
                                                                          category_index)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            cv2.imshow('object',cv2.resize(image_np,(800,600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                

            
    return (ymin,xmin, ymax, xmax,status)



def run_detection(frame):
    print('ModelLoaded')
    imagetorundetection = frame
    print('Image Loaded')
    status = False
    while status == False:
        (ymin,xmin, ymax, xmax,status) = crib_detector(imagetorundetection)
        print(ymin)
        print(xmin)
        print(ymax)
        print(xmax)

    return (ymin,xmin, ymax, xmax)
