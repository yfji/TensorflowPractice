# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:02:09 2018

@author: samsung
"""

import numpy as np
import cv2

'''
base_scale: img_raw_height/input_size
scale_abs: scale_multiplier*(input_size/img_raw_height)
'''
def aug_scale(image, base_scale, scale_range, keypoints):
    prob=np.random.random()
    scale_multiplier=(scale_range[1]-scale_range[0])*prob+scale_range[0]
    scale_abs=scale_multiplier/base_scale
    img_scale=cv2.resize(image, (0,0), fx=scale_abs,fy=scale_abs,interpolation=cv2.INTER_CUBIC)                 
    keypoints[:,0]*=scale_abs
    keypoints[:,1]*=scale_abs
    return img_scale

def aug_rotate(image, rot_max, keypoints):
    prob=np.random.random()
    angle=rot_max*(prob-0.5)
    rad=np.deg2rad(angle)
    center=(image.shape[0]*0.5, image.shape[1]*0.5)
    R=cv2.getRotationMatrix2D(center, angle, 1)
    cv2.rotatedRectangleIntersection

def aug_crop(image, size, center_perterb_max, keypoints):
    pass