# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 20:11:14 2018

@author: JYF
"""

import cv2
import numpy as np

image=cv2.imread('I:/Experiment/images/hand_me.jpg')
h,w,_=image.shape
print(w,h)

deg=30
rad=np.deg2rad(deg)

sin,cos=np.sin(rad),np.cos(rad)

R=np.asarray([[cos,sin],[-sin,cos]])
corners=np.asarray([[0,0],[w,0],[w,h],[0,h]])
print(corners)
print('\n')
t_corners=corners.dot(R.T).astype(np.int32)
#t_corners[:,0]-=t_corners[:,0].min()
#t_corners[:,1]-=t_corners[:,1].min()

print(t_corners)

x,y,nw,nh=cv2.boundingRect(t_corners.reshape(1,-1,2))

print(x,y,nw,nh)

R=cv2.getRotationMatrix2D((w/2,h/2), deg, 1)
R[0,2]+=nw/2-w/2
R[1,2]+=nh/2-h/2

print(R)

img_rot=cv2.warpAffine(image, R, (nw,nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128,128,128))
cv2.imshow('rot', img_rot)
cv2.waitKey()
cv2.destroyAllWindows()
