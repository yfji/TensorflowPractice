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

max_iter=30
stepvalues=[150000,240000,300000-1]
base_lr=0.00008066
rate_decay=0.01
snapshot=20000
display=20
save_summary=10

with tf.device('/gpu:1'):
    g=tf.Graph()
    with g.as_default():
        loader=loader.DataLoader(csv_path='J:/Downloads/fashionAI_key_points_train_20180227/train/Annotations/train.csv',
                                 dataset_root='J:/Downloads/fashionAI_key_points_train_20180227/train')
        
        datashape,labelshape=loader.get_shape()
        
        tensor_in=tf.placeholder(shape=datashape, dtype=tf.float32)
        label=tf.placeholder(shape=labelshape, dtype=tf.float32)
        
        tensor_out=network.make_cpm(tensor_in)
        
        diff2_stage1=tf.square(tf.subtract(tensor_out[0], label))
        diff2_stage2=tf.square(tf.subtract(tensor_out[1], label))
        
    #    with tf.name_scope('loss1'):
        loss_stage1=tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(diff2_stage1, axis=1),axis=1),axis=1))
    #    with tf.name_scope('loss2'):
        loss_stage2=tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(diff2_stage2, axis=1),axis=1),axis=1))
    #    loss_stage2=tf.nn.l2_loss(tf.subtract(tensor_out[1], label))
        
        joint_loss=tf.placeholder(tf.float32)
        tf.summary.scalar('loss_stage1',loss_stage1)
        tf.summary.scalar('loss_stage2',loss_stage2)
        tf.summary.scalar('loss',joint_loss)
        
    #####   strange learning rate
        global_step=tf.Variable(0,trainable=False,name='global_step')
        decay_step=tf.Variable(stepvalues[0],trainable=False,name='decay_step')
    
        learning_rate=tf.Variable(base_lr, name='lr')
        exp_lr=tf.train.exponential_decay(learning_rate,
                                          global_step=global_step,
                                          decay_steps=decay_step,
                                          decay_rate=rate_decay,
                                          staircase=True)
        
        global_adder=tf.placeholder(tf.int32)
        decay_value=tf.placeholder(tf.int32)
        lr=tf.placeholder(tf.float32)
        base_step=tf.placeholder(tf.int32)
        
        add_global=global_step.assign_add(global_adder)
        decay_step=decay_step.assign(decay_value)
        lr_init=learning_rate.assign(lr)
        step_init=global_step.assign(base_step)
        
    #####
    
        op_stg1=tf.train.AdamOptimizer(0.01).minimize(loss_stage1)
        op_stg2=tf.train.AdamOptimizer(0.01).minimize(loss_stage2)
        
        lr_summ=tf.placeholder(tf.float32)
        tf.summary.scalar('learning_rate',lr_summ)
        
        savedir='./models'
        logdir='./log'
        
        with tf.Session() as sess:
            saver=tf.train.Saver
            summary_writer=tf.summary.FileWriter(op.join(logdir,'fashion_summaries'),sess.graph)
            sum_op=tf.summary.merge_all()
            
            sess.run(tf.global_variables_initializer())
             
            index=0
            for iteration in range(max_iter):
                data_, label_=loader.load_minibatch()
                sess.run([op_stg1,op_stg2], feed_dict={tensor_in:data_,label:label_})
                loss1,loss2=sess.run([loss_stage1,loss_stage2], feed_dict={tensor_in:data_,label:label_})
                rate=sess.run(exp_lr,feed_dict={decay_value:stepvalues[index]})
                if iteration==stepvalues[index]:
                    sess.run(lr_init, feed_dict={lr:rate})
                    sess.run(decay_step, feed_dict={decay_value:stepvalues[index+1]-stepvalues[index]})
                    sess.run(step_init,feed_dict={base_step:0})
                    print('rate decay: %f'%rate)
                    index+=1
                if iteration%display==0:
                    print('[%d/%d]loss: %f, learning_rate: %f'%(iteration, max_iter, loss1+loss2, rate))
                if iteration%snapshot==0:
                    saver.save(sess, op.join(savedir,'fashion.ckpt-%d'%iteration))
                if iteration%save_summary==0:
                    sess.run(sum_op,feed_dict={tensor_in:data_,label:label_,joint_loss:loss1+loss2,lr_summ:rate})
                sess.run(add_global,feed_dict={global_adder:1})
        
