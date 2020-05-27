import numpy as np
from PIL import Image
import cv2
import logging

logging.basicConfig(filename='opencv.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

def main(img, LThreshold, num_erosion):
    logging.debug("LThreshold: %d, num_erosion: %d", LThreshold, num_erosion)
    y = np.array(img).reshape((800, 800)).astype(np.uint8)
    ret,img = cv2.threshold(y, LThreshold, 255, cv2.THRESH_BINARY)
    if num_erosion > 0:
        kernel = np.ones((5,5),np.uint8) 
        img = cv2.erode(img, kernel, num_erosion)
    return img.reshape((640000,)).tolist()