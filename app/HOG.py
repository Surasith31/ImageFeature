import cv2
import numpy as np

def hog_des(img):
    new_img = cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
    win_size = new_img.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bin = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,cell_size, num_bin)

    hog_descriptor = hog.compute(new_img)
    
    return hog_descriptor