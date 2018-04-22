# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 19:00:43 2018

@author: yfji
"""

import tensorflow as tf
import numpy as np
import network
import model_loader as ml
import cv2
import csv
import os.path as op

input_size=256
category_keypoints={
    'blouse':[0,1,2,3,4,5,6,9,10,11,12,13,14],
    'dress':[0,1,2,3,4,5,6,7,8,17,18],
    'skirt':[15,16,17,18],
    'outwear':[0,1,3,4,5,6,9,10,11,12,13,14],
    'trousers':[15,16,19,20,21,22,23]
}

def find_peaks(fmap, thresh=0.1):
    map_left = np.zeros(fmap.shape)
    map_left[1:,:] = fmap[:-1,:]
    map_right = np.zeros(fmap.shape)
    map_right[:-1,:] = fmap[1:,:]
    map_up = np.zeros(fmap.shape)
    map_up[:,1:] = fmap[:,:-1]
    map_down = np.zeros(fmap.shape)
    map_down[:,:-1] = fmap[:,1:]
    
    peaks_binary = np.logical_and.reduce((fmap>=map_left, fmap>=map_right, fmap>=map_up, fmap>=map_down, fmap > thresh))
    peaks = np.hstack((np.nonzero(peaks_binary)[1].reshape(-1,1), np.nonzero(peaks_binary)[0].reshape(-1,1))) # note reverse
    peaks_with_score = [(x[0],x[1]) + (fmap[x[1],x[0]],) for x in peaks]
    return peaks_with_score

def conv_keypoints(tensor_out, keypoints):
    keypoints_det=-1*np.ones((24,3),dtype=np.float32)
    ltx=1e4;lty=1e4;rbx=0;rby=0
    for i in range(24):
        heatmap=tensor_out[:,:,i]
        peaks=find_peaks(heatmap)
        if len(peaks)>0:
            peaks=sorted(peaks, key=lambda x:x[2], reverse=True)
            peak=peaks[0]
            raw_peak=[peak[0]*1.0,peak[1]*1.0,1.0]
            keypoints_det[i]=np.asarray(raw_peak)
            if i in keypoints:
                ltx=min(ltx,raw_peak[0])
                lty=min(lty,raw_peak[1])
                rbx=max(rbx,raw_peak[0])
                rby=max(rby,raw_peak[1])
    center=[0.5*(ltx+rbx),0.5*(lty+rby)]
    arr=np.zeros(24)
    arr[keypoints]=1
    keypoints_det[arr==0,0]=center[0]
    keypoints_det[arr==0,1]=center[1]
    keypoints_det[arr==0,2]=0
    return keypoints_det
    

dataset_root='/home/yfji/Workspace/Python/FashionAI/test_new/test/'
iteration=20000

with tf.Graph().as_default():
    tensor_in=tf.placeholder(shape=[1,input_size,input_size,3],dtype=tf.float32)
    
    tensor_out=network.make_cpm(tensor_in)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
        
    with tf.Session(config=config) as sess:
        model_loader=ml.ModelLoader(load_mode='ckpt',model_path='models/fashion.ckpt-%d'%iteration)
        model_loader.load_model(session=sess)
        
        with open('/home/yfji/benchmark/Keypoint/fashionAI_key_points_train_20180227/train/Annotations/train.csv','r') as tf:
            reader=csv.reader(tf)
            header=None
            for row in reader:
                header=list(row)
                break
                print(header)
        csv_name='pred_iter_%d_0.85.csv'%iteration
        rows=[]
        with open(op.join(dataset_root,'test.csv'), 'r') as f:
            reader=csv.reader(f)
            for row in reader:
                rows.append(list(row))
        num_samples=len(rows)
        dummy_header=rows[0]
        print(dummy_header)
        rows=rows[1:]
        with open(csv_name,'w') as pred_f:
            pred_writer=csv.writer(pred_f, dialect='excel')
            pred_writer.writerow(header)
            for ix,row in enumerate(rows):
                image_path=op.join(dataset_root, row[0])
                category=row[1]
                raw_image=cv2.imread(image_path)
                scale=0.85*input_size/max(1.0*raw_image.shape[0],1.0*raw_image.shape[1])
                image_scale=cv2.resize(raw_image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                
                input_image=128*np.ones((input_size,input_size,3),dtype=np.float32)
                input_image[:image_scale.shape[0],:image_scale.shape[1],:]=image_scale
                input_image=input_image[np.newaxis,:,:,:]/256.0-0.5
                
                output=np.squeeze(sess.run(tensor_out[1], feed_dict={tensor_in:input_image}))
                
                stride=1.0*input_size/output.shape[0]
                output=cv2.resize(output, (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
                
                keypoints_det=conv_keypoints(output, category_keypoints[category])
                keypoints_det[:,0:2]/=scale
                
                out_list=[]
                for k in range(24):
                    out_list.append('%d_%d_1'%(keypoints_det[k,0],keypoints_det[k,1]))
                pred_row=[row[0],category]+out_list
                pred_writer.writerow(pred_row)
                print('%d/%d'%(ix+1,num_samples))
            print('done')
                    