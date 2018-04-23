# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:05:21 2018

@author: JYF
"""

import numpy as np
import matplotlib.colors as mc
import matplotlib.pylab as plb

colors=['r','g','b','y']
markers=['+','o','x','-']

def load_data(num_clusters=4, sample_size=500, centers=None, cov_mat=None):
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
    
def visualize_region(X, Y, op=None, feed=None, session=None):
    print(Y[:10])
     
    pX=X[Y==0]
    nX=X[Y==1]

    plb.scatter(pX[:,0],pX[:,1],c=colors[0], marker=markers[0])
    plb.scatter(nX[:,0],nX[:,1],c=colors[1], marker=markers[1])
    
    nb_of_xs=200
    xs1=np.linspace(-1,8,num=nb_of_xs)
    xs2=np.linspace(-1,8,num=nb_of_xs)
    xx,yy=np.meshgrid(xs1,xs2)
    
    plane=np.zeros((nb_of_xs,nb_of_xs))
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            out=session.run(op, feed_dict={feed: [[xx[i,j],yy[i,j]]]})[0]
            plane[i,j]=out.argmax()
    cmap=mc.ListedColormap([mc.colorConverter.to_rgba('r',alpha=0.2),
                           mc.colorConverter.to_rgba('b',alpha=0.2)])
    plb.contourf(xx,yy,plane,cmap=cmap)
    plb.show()
    
    
