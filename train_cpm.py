# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:56:50 2018

@author: samsung
"""

import tensorflow as tf
import numpy as np
import network
import data_loader as loader

max_iter=300000
stepvalues=[150000,240000]

g=tf.Graph()
with g.as_default():
    loader=loader.DataLoader()
    
    batchshape=loader.get_batchshape()
    
    tensor_in=tf.placeholder(shape=batchshape, dtype=tf.float32)
    
    tensor_out=network.make_cpm(tensor_in)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data, label=loader.load_minibatch()
        out=sess.run(tensor_out, feed_dict={tensor_in:data})
        
        global_step=tf.Variable(0,trainable=False,name='global_step')
        init_lr=0.1
        lr=tf.train.exponential_decay(init_lr,
                                      global_step=global_step,
                                      decay_steps=10,
                                      decay_rate=0.5,
                                      staircase=True)
        add_global=global_step.assign_add(1)
        
        