# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:56:50 2018

@author: samsung
"""

import tensorflow as tf
import network
import data_loader as loader
import os.path as op
import numpy as np
import model_loader as ml
import os
os.environ['CUDA_VISIBLE_DEVICES']= '2'

#max_iter=30000
#stepvalues=[20000,25000,30000-1,30000]
#base_lr=0.00002066
#rate_decay=0.1
#snapshot=5000
max_iter=5000
stepvalues=[3000,5000]
base_lr=0.000002
rate_decay=0.1
snapshot=1000
display=20
save_summary=10

dataset_root='/home/yfji/benchmark/Keypoint/fashionAI_key_points_train_20180227/train/'

with tf.device('/gpu:2'):
    g=tf.Graph()
    with g.as_default():
        loader=loader.DataLoader(csv_path=op.join(dataset_root,'Annotations/train.csv'),
                                 dataset_root=dataset_root)
        model_loader=ml.ModelLoader(load_mode='ckpt',model_path='./models/fashion.ckpt-25000')
        
        datashape,labelshape=loader.get_shape()
        
        tensor_in=tf.placeholder(shape=datashape, dtype=tf.float32)
        label=tf.placeholder(shape=labelshape, dtype=tf.float32)
        
        tensor_out=network.make_cpm(tensor_in)
        
        diff2_stage1=tf.square(tf.subtract(tensor_out[0], label))
        diff2_stage2=tf.square(tf.subtract(tensor_out[1], label))
        
    #    with tf.name_scope('loss1'):
        loss_stage1=0.5*tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(diff2_stage1, axis=1),axis=1),axis=1))
    #    with tf.name_scope('loss2'):
        loss_stage2=0.5*tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(diff2_stage2, axis=1),axis=1),axis=1))
    #    loss_stage2=tf.nn.l2_loss(tf.subtract(tensor_out[1], label))
        
        joint_loss=tf.add(loss_stage1,loss_stage2)
        tf.summary.scalar('loss_stage1',loss_stage1)
        tf.summary.scalar('loss_stage2',loss_stage2)
        tf.summary.scalar('loss',joint_loss)
        
    #####   strange learning rate
        global_step=tf.Variable(0,trainable=False,name='global_step')
        decay_step=tf.Variable(stepvalues[0],trainable=False,name='decay_step')
    
        learning_rate=tf.Variable(base_lr, trainable=False, name='lr')
#        learning_rate=tf.get_variable(name='lrate', dtype=tf.float32, shape=[1], initializer=tf.constant_initializer(base_lr))
        exp_lr=tf.train.exponential_decay(learning_rate,
                                          global_step=global_step,
                                          decay_steps=decay_step,
                                          decay_rate=rate_decay,
                                          staircase=False)
        
        global_adder=tf.placeholder(tf.int32)
        decay_value=tf.placeholder(tf.int32)
        lr=tf.placeholder(tf.float32)
        base_step=tf.placeholder(tf.int32)
        
        add_global=global_step.assign_add(global_adder)
        decay=decay_step.assign(decay_value)
        lr_init=learning_rate.assign(lr)
        step_init=global_step.assign(base_step)
        
    #####
    
        op_stg1=tf.train.AdamOptimizer(exp_lr).minimize(loss_stage1)
        op_stg2=tf.train.AdamOptimizer(exp_lr).minimize(loss_stage2)
        
        lr_summ=tf.placeholder(tf.float32)
        lr_summ_v=tf.Variable(0.0, trainable=False, dtype=tf.float32)
        tf.summary.scalar('learning_rate',lr_summ_v.assign(lr_summ))
        
        savedir='./models_2'
        logdir='./log'
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            model_loader.load_model(session=sess)
            saver=tf.train.Saver()
            summary_writer=tf.summary.FileWriter(op.join(logdir,'fashion_summaries'),sess.graph)
            sum_op=tf.summary.merge_all()
             
            index=0
            g_steps=stepvalues[0]
            for iteration in range(max_iter):
                data_, label_=loader.load_minibatch()
                sess.run([op_stg1,op_stg2], feed_dict={tensor_in:data_,label:label_})
                loss1,loss2=sess.run([loss_stage1,loss_stage2], feed_dict={tensor_in:data_,label:label_})
                rate=sess.run(exp_lr)
                if iteration==stepvalues[index]:
                    g_steps=stepvalues[index+1]-stepvalues[index]
                    sess.run(lr_init, feed_dict={lr:rate})
                    sess.run(decay, feed_dict={decay_value:g_steps})
                    sess.run(step_init,feed_dict={base_step:0})
                    print('rate decay: %e'%rate)
                    index+=1
                if iteration>0 and iteration%display==0:
                    print('[%d/%d]loss: %f, learning_rate: %e'%(iteration, max_iter, loss1+loss2, rate))
                if iteration>0 and iteration%snapshot==0:
                    saver.save(sess, op.join(savedir,'fashion.ckpt-%d'%iteration))
                if iteration>0 and iteration%save_summary==0:
                    sess.run(sum_op,feed_dict={tensor_in:data_,label:label_,joint_loss:loss1+loss2,lr_summ:rate})
                sess.run(add_global,feed_dict={global_adder:1})
        
