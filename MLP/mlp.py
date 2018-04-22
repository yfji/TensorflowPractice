# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 13:46:34 2018

@author: JYF
"""

import tensorflow as tf
import numpy as np

class MLP:
    def __init__(self, sizes=None, ckpt_path=None, mode='TRAIN'):
        self.params_={}
        self.sizes_=sizes
        self.mode_=mode
        if mode!='TEST':
            for i in range(1,len(sizes)):
                with tf.variable_scope('l%d'%i):
                    self.params_['w%d'%i]=tf.get_variable('w', shape=[sizes[i-1],sizes[i]],dtype=tf.float32,
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
                    self.params_['b%d'%i]=tf.get_variable('b', shape=[sizes[i]], dtype=np.float32,
                                initializer=tf.constant_initializer(0.0))
        else:
            reader = tf.train.NewCheckpointReader(ckpt_path)  
            var_to_shape_map = reader.get_variable_to_shape_map()  
            for key in var_to_shape_map:  
                print(key)

                
    def net(self, batch):
        data=batch
        for i in range(1,len(self.sizes_)):
            data=tf.matmul(data, self.params_['w%d'%i])
            data=data+self.params_['b%d'%i]
            if i<len(self.sizes_)-1:
                data=tf.nn.relu(data)
        return data
        
        
        