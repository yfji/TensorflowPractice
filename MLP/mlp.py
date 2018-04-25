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
            self.sizes_=[]
            for key in var_to_shape_map:  
                print(key,var_to_shape_map[key])
                names=key.split('/')
                layer_name=names[0]
                var_name=names[1]
                var_shape=var_to_shape_map[key]
                with tf.variable_scope(layer_name):
                    layer_index=int(layer_name[1:])
#                    if layer_index not in self.sizes_:
#                        self.sizes_.append(layer_index)
                    self.params_['%s%d'%(var_name,layer_index)]=tf.get_variable(var_name, shape=var_shape, dtype=np.float32)
                    if 'w' in var_name:
                        if len(self.sizes_)==0:
                            self.sizes_=[var_shape[0],var_shape[1]]
                        else:
                            self.sizes_.append(var_shape[1])
            print(self.sizes_)
                
    def net(self, batch):
        data=batch
        for i in range(1,len(self.sizes_)):
            data=tf.matmul(data, self.params_['w%d'%i])
            data=data+self.params_['b%d'%i]
            if i<len(self.sizes_)-1:
                data=tf.nn.relu(data)
        return data
