from stitch_image import Stitcher
import cv2
import numpy as np
images=[]
images.append( cv2.imread('./images/1.jpeg'))
images.append( cv2.imread('./images/2.jpeg'))
images.append( cv2.imread('./images/3.jpeg'))
images.append( cv2.imread('./images/4.jpeg'))
images.append( cv2.imread('./images/5.jpeg'))
images.append( cv2.imread('./images/6.jpeg'))

Stitcher(images)