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
    scale_abs=scale_multiplier*base_scale
    img_scale=cv2.resize(image, (0,0), fx=scale_abs,fy=scale_abs,interpolation=cv2.INTER_CUBIC)                 
    keypoints[:,0]*=scale_abs
    keypoints[:,1]*=scale_abs
    return img_scale

def aug_rotate(image, rot_max, keypoints):
    prob=np.random.random()
    angle=rot_max*(prob-0.5)
    rad=np.deg2rad(angle)
    h,w,_=image.shape
    
    sin,cos=np.sin(rad),np.cos(rad)
    R=np.asarray([[cos,sin],[-sin,cos]])
    corners=np.asarray([[0,0],[w,0],[w,h],[0,h]])
    t_corners=corners.dot(R.T).astype(np.int32)
    
    keypoints[:,0:2]=keypoints[:,0:2].dot(R.T)
    
    x,y,nw,nh=cv2.boundingRect(t_corners.reshape(1,-1,2))

    center=(w*0.5, h*0.5)
    R=cv2.getRotationMatrix2D(center, angle, 1)
    R[0,2]+=nw/2-w/2
    R[1,2]=nh/2-h/w
    img_rot=cv2.warpAffine(image, R, (nw,nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128,128,128))
    
    keypoints[:,0]+=R[0,2]
    keypoints[:,1]+=R[1,2]
    return img_rot

def aug_crop(image, net_input_size, center_perterb_max, keypoints):
    probs=np.random.random(size=2)
    mw,mh=int(image.shape[1]/2-0.5), int(image.shape[0]/2-0.5)
    dx,dy=int((probs[0]-0.5)*center_perterb_max*2), int((probs[1]-0.5)*center_perterb_max*2)
    cx,cy=int(net_input_size/2),int(net_input_size/2)
    
    crop_image=128*np.ones((net_input_size,net_input_size,3),dtype=np.float32)
    
    startx1=max(0,cx-mw+dx)
    starty1=max(0,cy-mh+dy)
    endx1=min(net_input_size, cx+mw+dx)
    endy1=min(net_input_size, cy+mh+dy)
    startx2=max(0,mw-dx-cx)
    starty2=max(0,mh-dy-cy)
    endx2=min(2*mw,mw-dx+cx)
    endy2=min(2*mh,mh-dy+cy)
    
    keypoints[:,0]+=(startx1-startx2)
    keypoints[:,1]+=(starty1-starty2)
    
#    startx1,startx2,starty1,starty2,endx1,endx2,endy1,endy2=list(map(int,[startx1,startx2,starty1,starty2,endx1,endx2,endy1,endy2]))
    crop_image[starty1:endy1,startx1:endx1,:]=image[starty2:endy2,startx2:endx2,:]
    return crop_image
    
