# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:33:59 2018
@author: JYF
"""

import tensorflow as tf
import numpy as np
import mlp
import data_loader as dl

num_classes=2
num_samples=2000
centers={'0':[[1.,1.],[5.,4.]],'1':[[1.,4.],[5.,1.]]}
cov_mat=np.asarray([[0.1,0.05],[0.05,0.1]])

dataset, labels=dl.load_data(num_clusters=4, sample_size=num_samples,centers=centers, cov_mat=cov_mat)

batch_size=16
max_iter=8000
lr=0.1
decay_ratio=0.333
cur_index=0
display=20
snapshot=1000
step_index=0
stepvalues=[5000,7000,8000]
g_steps=stepvalues[0]

random_order=np.random.permutation(np.arange(num_samples))

with tf.device('/cpu:0'):
    tf.reset_default_graph()
    
    base_lr=tf.placeholder(tf.float32)
    cur_step=tf.placeholder(tf.float32)
    decay_steps=tf.placeholder(tf.float32)
    
    lr_decay=tf.train.exponential_decay(learning_rate=base_lr, 
                                        global_step=cur_step,
                                        decay_steps=decay_steps,
                                        decay_rate=decay_ratio)
    
    tensor_in=tf.placeholder(tf.float32, shape=[batch_size,dataset.shape[1]])
    batch_labels=tf.placeholder(tf.int32, shape=[batch_size,num_classes])
    
    nn=mlp.MLP(sizes=[2,200,2])
    
    tensor_out=nn.net(tensor_in)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_labels, logits=tensor_out))
    
    exp_lr=tf.placeholder(tf.float32)
    loss_summ=tf.placeholder(tf.float32)

    tf.summary.scalar('loss',loss_summ)
    tf.summary.scalar('learning_rate',exp_lr)
    summ_op=tf.summary.merge_all()
    
    optimizer=tf.train.GradientDescentOptimizer(exp_lr).minimize(loss)
    
    with tf.Session() as sess:
        saver=tf.train.Saver(max_to_keep=20)
        writer=tf.summary.FileWriter('log/mlp',sess.graph)
        sess.run(tf.global_variables_initializer())
        step=0
        for i in range(max_iter):
            xs=dataset[random_order[cur_index:cur_index+batch_size]]
            ys=labels[random_order[cur_index:cur_index+batch_size]].astype(np.int32)
            
            one_hots=np.zeros((batch_size,num_classes),dtype=np.int32)
            for k in range(one_hots.shape[0]):
                one_hots[k,ys[k]]=1
            #learning rate decay
            rate=sess.run(lr_decay,feed_dict={base_lr:lr,cur_step:step,decay_steps:g_steps})
            _loss,_=sess.run([loss,optimizer], feed_dict={tensor_in:xs, batch_labels:one_hots, exp_lr:rate})
            
            summ_str=sess.run(summ_op, feed_dict={loss_summ:_loss,exp_lr:rate})
            writer.add_summary(summ_str,i)
            
            if i%display==0:
                print('[%d/%d] loss: %f, learn rate: %e'%(i,max_iter,_loss, rate))
            if i>0 and i%snapshot==0:
                saver.save(sess, 'models/mlp.ckpt-%d'%i)
            if i==stepvalues[step_index]:
                print('learn rate decay: %e'%rate)
                lr=rate
                step=0
                g_steps=stepvalues[step_index+1]-stepvalues[step_index]
                step_index+=1
            step+=1
                
            cur_index+=batch_size
            if cur_index+batch_size>=num_samples:
                random_order=np.random.permutation(np.arange(num_samples))
                cur_index=0
        saver.save(sess, 'models/mlp.ckpt-%d'%max_iter)
        print('done')

dl.visualize(dataset, labels)
