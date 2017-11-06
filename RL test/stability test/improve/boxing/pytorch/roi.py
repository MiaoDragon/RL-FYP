#ref: https://pythonprogramming.net/lane-region-of-interest-python-plays-gta-v/
import numpy as np
import cv2

def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked
