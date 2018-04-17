# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:00:00 2018

@author: samsung
"""

import numpy as np
import data_augment as da
import os.path as op
import csv

class DataLoader:
    def __init__(self, csv_path, dataset_root=''):
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
        self.angle_max=20
        self.shuffle()
    
    def shuffle(self):
        self.random_order_=np.random.permutation(np.arange(self.num_samples_))
        self.cur_index_=0
        
    def get_batchshape(self):
        return [self.minibatch_,self.net_input_size_, self.net_input_size_,3]
        
    def load_minibatch(self,probs=None): #probs for scale, rotation, flip, crop
        for i in range(self.cur_index_,min(self.cur_index_+self.minibatch_,self.num_samples_)):
            row=self.dataset_[self.random_order_[i]]
            image_path=op.join(self.dataset_root_,row[0])
            kpstrs=row[2:]
            keypoints=np.asarray((24,3),dtype=np.float32)
            for k, kpstr in enumerate(kpstrs):
                kps=list(map(float, kpstr.split('_')))
                keypoints[k]=np.asarray(kps)
            image=cv2.imread(image_path)
            image=da.aug_scale(image)
            image=da.aug_rotate(image)
            image
        return None
        
    def putGaussianMap(self):
        pass
    
    def generateLabel(self):
        pass
