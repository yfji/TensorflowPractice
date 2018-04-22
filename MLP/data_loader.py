# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:05:21 2018

@author: JYF
"""

import numpy as np
import matplotlib.pylab as plb

cov_mat=np.asarray([[0.2,0.],[0.,0.2]])
colors=['r','g','b','y']

def load_data(num_clusters=4, sample_size=500, centers=None):
    samples_per_class=int(sample_size/num_clusters)
    trainX=np.zeros((0,2),dtype=np.float32)
    trainY=np.zeros(0, dtype=np.int32)
    for k in centers:
        clusters=centers[k]
        for center in clusters:
            X=np.random.multivariate_normal(center, cov_mat, samples_per_class)
            Y=np.tile(k, samples_per_class)
            trainX=np.vstack((trainX,X))
            trainY=np.concatenate((trainY, Y))
    print(trainX.shape)
    print(trainY.shape)
    return trainX,trainY

def visualize(trainX, trainY):
    color=[colors[int(i)] for i in trainY]
    plb.scatter(trainX[:,0],trainX[:,1],c=color)
    plb.xlabel('x')
    plb.ylabel('y')
    plb.show()