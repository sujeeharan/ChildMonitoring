import collections 
import numpy as np

def get_coordinates_from_tensor(image,
                                boxes,
                                classes,
                                scores,
                                category_index,
                                instance_masks=None,
                                keypoints=None,
                                use_normalized_coordinates=False,
                                max_boxes_to_draw=20,
                                minimum_xdetect_percentage=90,
                                agnostic_mode=False,

                                line_thickness=4):
  
  box_to_display_str_map = collections.defaultdict(list)
  box_to_instance_masks_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  ymin=0
  ymax = 0
  xmin=0
  xmax=0
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):

      if classes[i] in category_index.keys():
          class_name = category_index[classes[i]]['name']
      if class_name=='crib':
          if scores is not None or scores[i] > minimum_detect_percentage:
              
              for box,color in box_to_color_map.items():
                  ymin,xmin,ymax,xmax = box
              return (ymin,xmin,ymax,xmax,True)
      else:
          return (None,None,None,None,False)

