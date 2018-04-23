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
num_samples=1200
centers={'0':[[2.5,2.]],'1':[[0.5,5.],[6.,1.6]]}
cov_mat=np.asarray([[0.3,0.2],[0.2,0.3]])

dataset, labels=dl.load_data(num_clusters=3, sample_size=num_samples,centers=centers, cov_mat=cov_mat)
dl.visualize(dataset, labels)

with tf.device('/cpu:0'):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tensor_in=tf.placeholder(tf.float32, shape=[None,2])
        nn=mlp.MLP(sizes=None, ckpt_path=model_path, mode='TEST')
        tensor_out=nn.net(tensor_in)
        saver=tf.train.Saver()
        
        with tf.Session() as sess:        
            saver.restore(sess, model_path)
            one_hot=sess.run(tensor_out,feed_dict={tensor_in:dataset})
            print(one_hot.shape)
            preds=np.zeros(len(dataset), np.int32)
            for i in range(len(dataset)):
                preds[i]=np.argmax(one_hot[i])
            dl.visualize_region(dataset, preds, tensor_out, tensor_in, sess)

        # create new variables for the graph
        with tf.variable_scope('l3',reuse=None):
            v=tf.get_variable('v',shape=[2,10],dtype=tf.float32, initializer=tf.constant_initializer(1))
        saver=tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, 'models/mlp-addvar1.ckpt')
            reader = tf.train.NewCheckpointReader('models/mlp-addvar1.ckpt')  
            var_to_shape_map = reader.get_variable_to_shape_map()  
            for key in var_to_shape_map:  
                print(key,var_to_shape_map[key])
