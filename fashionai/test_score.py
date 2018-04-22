# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:27:13 2018

@author: yfji
"""

import tensorflow as tf
import numpy as np
import cv2
import os.path as op
import network
import model_loader as ml
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'


input_size=256

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
    
category_keypoints={
    'blouse':[0,1,2,3,4,5,6,9,10,11,12,13,14],
    'dress':[0,1,2,3,4,5,6,7,8,17,18],
    'skirt':[15,16,17,18],
    'outwear':[0,1,3,4,5,6,9,10,11,12,13,14],
    'trousers':[15,16,19,20,21,22,23]
}

def criterion(category, keypoints_gt, keypoints_det):
    norm_dist=0
    if category in ['blouse','outwear','dress']:
        pt1=keypoints_gt[5,:2]
        pt2=keypoints_gt[6,:2]
        norm_dist=euclideanDistance(pt1,pt2)
    elif category in ['trousers','skirt']:
        pt1=keypoints_gt[15,:2]
        pt2=keypoints_gt[16,:2]
        norm_dist=euclideanDistance(pt1,pt2)
    else:
        raise Exception('Unknown type')
    scores=[]
    if norm_dist==0:
        return []
    for k in range(keypoints_gt.shape[0]):
        if keypoints_gt[k,-1]==1:
            dist=euclideanDistance(keypoints_gt[k],keypoints_det[k])
            scores.append(1.0*dist/norm_dist)
    return scores
    
def euclideanDistance(pt1, pt2):
    dist_vec=np.subtract(pt1,pt2)
    return np.sqrt(np.sum(dist_vec**2))

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
    
    
def conv_keypoints_batch(tensor_out, keypoints):
    batch_size=tensor_out.shape[0]
    batch_keypoints=[]
    for b in range(batch_size):
        tensor=tensor_out[b]
        keypoints_det=-1*np.ones((24,3),dtype=np.float32)
        
        ltx=1e4;lty=1e4;rbx=0;rby=0
        for i in range(24):
            heatmap=tensor[:,:,i]
#            heatmap=heatmap[:,:,np.newaxis]
            peaks=find_peaks(heatmap)
            if len(peaks)>0:
                peaks=sorted(peaks, key=lambda x:x[2], reverse=True)
                peak=peaks[0]
                raw_peak=[peak[0]*1.0,peak[1]*1.0,1.0]
                keypoints_det[i]=np.asarray(raw_peak)
                if i in keypoints[b]:
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
        batch_keypoints.append(keypoints_det)
    return batch_keypoints
    
image_root='/home/yfji/benchmark/Keypoint/fashionAI_warm_up_train_20180222/train/'
test_batch=1

with tf.device('/gpu:0'):
    with tf.Graph().as_default():
        data=tf.placeholder(shape=[test_batch,input_size,input_size,3],dtype=tf.float32)
        
        tensor_out=network.make_cpm(data)
        
        rows=[]
        csv_name='/home/yfji/benchmark/Keypoint/fashionAI_warm_up_train_20180222/train/Annotations/annotations.csv'
        with open(csv_name,'r') as f:
            reader=csv.reader(f)
            for row in reader:
                rows.append(list(row))
        header=rows[0]
        rows=rows[1:]
        num_samples=len(rows)
        print(header, num_samples)
        
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        
        with tf.Session(config=config) as sess:
            model_loader=ml.ModelLoader(load_mode='ckpt',model_path='models/fashion.ckpt-20000')
            model_loader.load_model(session=sess)
            
            score_sum=0.0
            cnt=0
            random_order=np.random.permutation(np.arange(num_samples))
            for i in range(0, num_samples, test_batch):
                if i>=5000:
                    break
                tensor_in=np.zeros((test_batch, input_size, input_size, 3), dtype=np.float32)
                cate_keypoints=[]
                categories=[]
                images=[]
                scales=[]
                keypoints_gts=-1*np.ones((test_batch,24,3),dtype=np.float32)
                for j in range(test_batch):
                    row=rows[random_order[i+j]]
                    image_path=op.join(image_root, row[0])
                    category=row[1]
                    categories.append(category)
                    for k, kpstr in enumerate(row[2:]):
                        kps=kpstr.split('_')
                        keypoints_gts[j,k]=np.asarray(list(map(float,kps)))
                    cate_keypoints.append(category_keypoints[category])
                    raw_image=cv2.imread(image_path)
                    images.append(raw_image)
                    scale=0.9*input_size/max(1.0*raw_image.shape[0],1.0*raw_image.shape[1])
                    scales.append(scale)
                    image_scale=cv2.resize(raw_image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    input_image=128*np.ones((input_size,input_size,3),dtype=np.float32)
                    input_image[:image_scale.shape[0],:image_scale.shape[1],:]=image_scale
                    input_image=(input_image-128)*1.0/256.0
                    tensor_in[j]=input_image
                output=sess.run(tensor_out[1], feed_dict={data:tensor_in})
                stride=input_size/output.shape[1]
                for j in range(test_batch):
                    out=cv2.resize(output[j], (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
                    keypoints_det=conv_keypoints(out, cate_keypoints[j])
                    keypoints_det[:,0:2]/=scales[j]
    #                for k in range(24):
    #                    if k not in category_keypoints[categories[j]]:
    #                        color=(255,0,0)
    #                    else:
    #                        color=(0,255,0)
    #                        cv2.circle(images[j], (int(keypoints_det[k,0]),int(keypoints_det[k,1])), 6, color,-1)
    #                cv2.imshow('pred', images[j])
    #                cv2.waitKey()
    #                cv2.destroyAllWindows()
                
                    scores=criterion(categories[j], keypoints_gts[j],keypoints_det)
                    score_avg=0
                    if len(scores)>0:
                        score_avg=1.0*sum(scores)/len(scores)
                        cnt+=1
                    print('[%d/%d]:%f'%(i+j,num_samples,score_avg))
                    score_sum+=score_avg
            print('Total score: %f'%(score_sum/cnt))
            print('done')