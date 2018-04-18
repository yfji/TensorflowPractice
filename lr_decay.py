# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:19:40 2018

@author: samsung
"""

import tensorflow as tf

global_step=tf.Variable(0,trainable=False,name='global_step')
step=tf.Variable(1)
init_lr=0.1
lr=tf.train.exponential_decay(init_lr,
                              global_step=global_step,
                              decay_steps=10,
                              decay_rate=0.5,
                              staircase=False)

add_global=global_step.assign_add(1)
#add_global=tf.add(global_step,step)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('0\t',sess.run(lr))
    for i in range(20):
        add,rate=sess.run([add_global,lr])
#        print('global step: ',global_step.eval())
        print(add,'\t',rate)
  
"""
# -*- coding: utf-8 -*-

import tensorflow as tf

max_iter=100
stepvalues=[20,30,99,0]

global_step=tf.Variable(0,trainable=False,name='global_step')
decay_step=tf.Variable(stepvalues[0],trainable=False,name='global_step')

init_lr=tf.Variable(0.1, name='init_lr')

lr=tf.train.exponential_decay(init_lr,
                              global_step=global_step,
                              decay_steps=decay_step,
                              decay_rate=0.5,
                              staircase=False)
global_adder=tf.placeholder(tf.int32)
decay_flag=tf.placeholder(tf.int32)
base_lr=tf.placeholder(tf.float32)
base_step=tf.placeholder(tf.int32)

add_global=global_step.assign_add(global_adder)
decay_step=decay_step.assign(decay_flag)
lr_init=init_lr.assign(base_lr)
step_init=global_step.assign(base_step)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('0\t',sess.run(lr))
    index=0
    for i in range(max_iter):
        rate=sess.run(lr,feed_dict={decay_flag:stepvalues[index]})
        if i==stepvalues[index]:
            sess.run(lr_init, feed_dict={base_lr:rate})
            sess.run(decay_step, feed_dict={decay_flag:stepvalues[index+1]-stepvalues[index]})
            sess.run(step_init,feed_dict={base_step:0})
            print(i,'\t',rate)
            index+=1
        add=sess.run(add_global,feed_dict={global_adder:1})
"""
