# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:40:40 2018

@author: samsung
"""

import data_loader as dl

loader=dl.DataLoader(csv_path='E:/yfji/benchmark/fashionai/train/Annotations/train.csv',
                         dataset_root='E:/yfji/benchmark/fashionai/train')

for i in range(10):
    data_,label_=loader.load_minibatch()
print('done')