# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:40:40 2018

@author: samsung
"""

import data_loader as dl
import os.path as op

dataset_root='/home/yfji/benchmark/Keypoint/fashionAI_key_points_train_20180227/train/'
loader=dl.DataLoader(csv_path=op.join(dataset_root,'Annotations/train.csv'),
                         dataset_root=dataset_root)

for i in range(10):
    data_,label_=loader.load_minibatch()
print('done')
