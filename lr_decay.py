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