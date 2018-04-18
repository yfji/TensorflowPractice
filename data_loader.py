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
        self.net_input_size_=368
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
        self.angle_max_=20
        self.target_scale_=0.8
        self.scale_range_=[0.8,1.1]
        self.stride_=8
        self.num_parts_=24
        self.label_channels_=25
        self.sigma_=7.0
        self.visualize_=True
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
        
        for i in range(self.cur_index_,min(self.cur_index_+self.minibatch_,self.num_samples_)):
            row=self.dataset_[self.random_order_[i]]
            image_path=op.join(self.dataset_root_,row[0])
            image_names[i]=image_path
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
            image=da.aug_crop(image, self.net_input_size_, self.center_perterb_max_, keypoints_gt[i])
            imagedata[i-self.cur_index_,:,:,:]=(image-128)/256.0
            if self.visualize_:
                if self.image_index_<100:
                    cv2.imwrite(op.join(self.savedir,'sample_%d.jpg'%self.image_index_),image)
                self.image_index_+=1

        for i in range(self.minibatch_):
            self.putGaussianMap(imagelabel[i], keypoints_gt[i])
        self.cur_index_+=self.minibatch_
        if self.cur_index_>=self.num_samples_:
            self.shuffle()
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
