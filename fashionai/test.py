import tensorflow as tf
import numpy as np
import network
import model_loader as ml
import cv2

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
    
def euclideanDistance(pt1, pt2):
    dist_vec=np.subtract(pt1,pt2)
    return np.sqrt(np.sum(dist_vec**2))

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
    
    
with tf.Graph().as_default():
    tensor_in=tf.placeholder(shape=[1,input_size,input_size,3],dtype=tf.float32)
    
    tensor_out=network.make_cpm(tensor_in)
    
    with tf.Session() as sess:
        model_loader=ml.ModelLoader(load_mode='ckpt',model_path='models/fashion.ckpt-10000')
        model_loader.load_model(session=sess)
        

        category='trousers'
        img_path='/home/yfji/Workspace/Python/FashionAI/test/Images/%s/0fe3b8126aed0858573ed0395adbca2b.jpg'%category
#        category='blouse'
#        img_path='blouse1.jpg'
        raw_image=cv2.imread(img_path)
        scale=1.0*input_size/max(1.0*raw_image.shape[0],1.0*raw_image.shape[1])
        print('scale',scale)
        image_scale=cv2.resize(raw_image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        input_image=128*np.ones((input_size,input_size,3),dtype=np.float32)
        input_image[:image_scale.shape[0],:image_scale.shape[1],:]=image_scale
        input_image=input_image[np.newaxis,:,:,:]/256.0-0.5
        
        output=np.squeeze(sess.run(tensor_out[1], feed_dict={tensor_in:input_image}))
        
        stride=1.0*input_size/output.shape[0]
        output=cv2.resize(output, (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        
        keypoints=category_keypoints[category]
        keypoints_det=-1*np.ones((24,3),dtype=np.float32)
    
        ltx=1e4;lty=1e4;rbx=0;rby=0
        for i in range(24):
            heatmap=output[:,:,i]
            heatmap=heatmap[:,:,np.newaxis]
            peaks=find_peaks(heatmap)
            if len(peaks)>0:
                peaks=sorted(peaks, key=lambda x:x[2], reverse=True)
                peak=peaks[0]
                raw_peak=[peak[0]*1.0/scale,peak[1]*1.0/scale,1.0]
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
        
        for i in range(24):
            if i not in category_keypoints[category]:
                color=(255,0,0)
            else:
                color=(0,255,0)
            cv2.circle(raw_image, (int(keypoints_det[i,0]),int(keypoints_det[i,1])), 6, color,-1)
        cv2.imshow('pred', raw_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    
