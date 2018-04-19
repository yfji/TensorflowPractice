# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:00:00 2018

@author: samsung
"""

import numpy as np
import data_augment as da
import os.path as op
import csv
import cv2

class DataLoader:
    def __init__(self, csv_path='', dataset_root=''):
        self.minibatch_=10
        self.net_input_size_=256
        self.csv_name_=csv_path
        self.dataset_root_=dataset_root
        self.dataset_=[]
        with open(self.csv_name_,'r') as f:
            reader=csv.reader(f)
            for row in reader:
                self.dataset_.append(list(row))
        self.header_=self.dataset_[0]
        self.dataset_=self.dataset_[1:]
        self.num_samples_=len(self.dataset_)
        self.cur_index_=0
        self.center_perterb_max_=20
        self.angle_max_=12
        self.target_scale_=0.9
        self.scale_range_=[0.8,1.2]
        self.stride_=8
        self.num_parts_=24
        self.label_channels_=25
        self.sigma_=7.0
        self.visualize_=False
        self.savedir='./visualize'
        self.image_index_=0
        self.shuffle()
    
    def shuffle(self):
        self.random_order_=np.random.permutation(np.arange(self.num_samples_))
        self.cur_index_=0
        
    def get_shape(self):
        return [self.minibatch_,self.net_input_size_, self.net_input_size_,3],[self.minibatch_,self.net_input_size_/self.stride_,self.net_input_size_/self.stride_,self.num_parts_+1]
        
    def load_minibatch(self): #probs for scale, rotation, flip, crop
        label_side=int(self.net_input_size_/self.stride_)
        imagedata=np.zeros((self.minibatch_,self.net_input_size_, self.net_input_size_,3), dtype=np.float32)
        imagelabel=np.zeros((self.minibatch_,label_side,label_side,self.label_channels_),dtype=np.float32)
        image_names=['']*self.minibatch_
        keypoints_gt=np.zeros((self.minibatch_,24,3), dtype=np.float32)
        
        for i in range(self.cur_index_,self.cur_index_+self.minibatch_):
            row=self.dataset_[self.random_order_[i]]
            image_path=op.join(self.dataset_root_,row[0])
            image_names[i-self.cur_index_]=image_path
            kpstrs=row[2:]
            keypoints=np.zeros((24,3),dtype=np.float32)
            for k, kpstr in enumerate(kpstrs):
                kps=list(map(float, kpstr.split('_')))
                keypoints[k]=np.asarray(kps)
            keypoints_gt[i-self.cur_index_]=keypoints
        for i,image_path in enumerate(image_names):
            image=cv2.imread(image_path)
            base_scale=self.target_scale_/(image.shape[0]/self.net_input_size_)
            image=da.aug_scale(image, base_scale, self.scale_range_, keypoints_gt[i])
            image=da.aug_rotate(image, self.angle_max_, keypoints_gt[i])
            labeled_index=keypoints_gt[i,:,-1]!=-1   
            center=np.mean(keypoints_gt[i,labeled_index,:2],axis=0) 
            image=da.aug_crop(image, center, self.net_input_size_, self.center_perterb_max_, keypoints_gt[i])
            imagedata[i,:,:,:]=(image-128)/256.0

        for i in range(self.minibatch_):
            self.putGaussianMap(imagelabel[i], keypoints_gt[i])
        if self.visualize_:
            for i in range(self.minibatch_):
                g_map=imagelabel[i,:,:,-1]
                g_map=cv2.resize(g_map, (0,0), fx=self.stride_,fy=self.stride_,interpolation=cv2.INTER_CUBIC)
                raw_image=(imagedata[i]*256.0+128).astype(np.uint8)
                vis_img=self.visualize(raw_image,g_map)
                if self.image_index_<100:
                    cv2.imwrite(op.join(self.savedir,'sample_%d.jpg'%self.image_index_),vis_img)
                self.image_index_+=1
        self.cur_index_+=self.minibatch_
        if self.cur_index_+self.minibatch_>self.num_samples_:
            self.shuffle()
        print('batch loaded')
        return imagedata,imagelabel
        
    def putGaussianMap(self, label, keypoints, sigma=7.0):
        assert(label.shape[2]==keypoints.shape[0]+1)
        start = self.stride_ / 2.0 - 0.5
        for i in range(label.shape[2]-1):    #[h,w,c]
            kp=keypoints[i]
            if kp[-1]!=-1:
                for y in range(label.shape[0]):
                    for x in range(label.shape[1]):
                        yy = start + y * self.stride_
                        xx = start + x * self.stride_
                        dis = ((xx - kp[0]) * (xx - kp[0]) + (yy - kp[1]) * (yy - kp[1])) / 2.0 / sigma / sigma
                        if dis > 4.6052:
                            continue
                        label[y,x,i] += np.exp(-dis)
                        label[y,x,i]=min(1,label[y,x,i])
        label[:,:,-1]=np.max(label[:,:,:-1],axis=2)
        
    def visualize(self,image, g_map):
        heatmap_bgr=np.zeros(image.shape, dtype=np.uint8)
        for i in range(heatmap_bgr.shape[0]):
            for j in range(heatmap_bgr.shape[1]):
                heatmap_bgr[i,j,[2,1,0]]=self.getJetColor(1-g_map[i,j],0,1)
        out_image=cv2.addWeighted(image, 0.7, heatmap_bgr, 0.3, 0).astype(np.uint8)
        return out_image
    
    def getJetColor(self, v, vmin, vmax):
        c = np.zeros((3))
        if (v < vmin):
            v = vmin
        if (v > vmax):
            v = vmax
        dv = vmax - vmin
        if (v < (vmin + 0.125 * dv)): 
            c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
        elif (v < (vmin + 0.375 * dv)):
            c[0] = 255
            c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
        elif (v < (vmin + 0.625 * dv)):
            c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
            c[1] = 255
            c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
        elif (v < (vmin + 0.875 * dv)):
            c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
            c[2] = 255
        else:
            c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5                      
        return c
            
