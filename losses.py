# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf

g=tf.Graph()

with g.as_default():
    labels=[[0,0,1],[0,1,0]]
    logits=[[2,0.5,6],[0.1,0,3]]
    
    logits_scaled=tf.nn.softmax(logits)
    
    loss1=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
    loss2=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits_scaled)
    
    loss3=-tf.reduce_sum(labels*tf.log(logits_scaled),axis=1)
    loss4=tf.reduce_sum(tf.square(tf.subtract(logits,labels)),axis=1)

    with tf.Session() as sess:
        print("scaled: ",sess.run(logits_scaled))
        print("loss1: ",sess.run(loss1))
        print("loss2: ",sess.run(loss2))
        print("loss3: ",sess.run(loss3))
        print("loss4: ",sess.run(loss4))
    
    print(sess.graph)
    print(tf.get_default_graph())

print(sess.graph)
print(tf.get_default_graph())
