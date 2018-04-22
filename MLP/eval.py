# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 15:35:47 2018

@author: JYF
"""

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plb
import data_loader as dl
import mlp

iteration=8000
model_path='models/mlp.ckpt-%d'%iteration
num_classes=2
num_samples=2000
centers={'0':[[1.,1.],[5.,4.]],'1':[[1.,4.],[5.,1.]]}

dataset, labels=dl.load_data(num_clusters=4, sample_size=num_samples,centers=centers)
dl.visualize(dataset, labels)

with tf.device('/cpu:0'):
    with tf.Graph().as_default():
        tensor_in=tf.placeholder(tf.float32, shape=[None,2])
        saver=tf.train.Saver()
        with tf.Session() as sess:        
            
            saver.restore(sess, model_path)
        
        v=tf.get_variable('v',shape=[10,10],dtype=tf.float32, initializer=tf.constant_initializer(1))
    
        nn=mlp.MLP(sizes=None, ckpt_path=model_path, mode='TEST')
        tensor_out=nn.net(tensor_in)
    
        preds=sess.run(tensor_out,feed_dict={tensor_in:dataset})
        
        

        
            
