# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:47:47 2018

@author: samsung
"""
import tensorflow as tf

def make_net_dict():
    #kernel: [h,w,c,n]
    #data: [n,h,w,c]
    feature = [{'conv1_1': [3,3,4,64]}, {'conv1_2': [3,3,64, 64]}, {'pool1': [1,2,2,1]},
            {'conv2_1': [3,3,64, 128]}, {'conv2_2': [3,3,128, 128]}, {'pool2': [1,2, 2, 1]},
            {'conv3_1': [3,3,128, 256]}, {'conv3_2': [3,3,256, 256]}, {'conv3_3': [3,3,256, 256]}, {'conv3_4': [3,3,256, 256]}, {'pool3': [1,2, 2, 1]},
            {'conv4_1': [3,3,256, 512]}, {'conv4_2': [3,3,512, 512]}, {'conv4_3_CPM': [3,3,512, 256]}, {'conv4_4_CPM': [3,3,256,128]}]


    stage1 = [{'conv5_1_CPM': [3,3,128, 128]},{'conv5_2_CPM': [3,3,128, 128]},{'conv5_3_CPM': [3,3,128, 128]},
             {'conv5_4_CPM': [1,1,128, 512], 'conv5_5_CPM': [1,1,512,25]}]


    stage2 = [{'Mconv1': [7,7,128+25, 128]}, {'Mconv2': [7,7,128, 128]},
              {'Mconv3': [7,7,128, 128]},{'Mconv4': [7,7,128, 128]},
              {'Mconv5': [7,7,128, 128]},
              {'Mconv6': [1,1,128, 128], 'Mconv7':[1,1,128,25]}
              ]
    predict_layers_stage1 = [{'conv5_5_CPM': [1,1,512, 25]}]

    predict_layers_stage2 = [{'conv5_5_CPM': [1,1,128, 25]}]
    
    concats=[['conv4_4_CPM','conv5_5_CPM','Mconv1']]

    net_dict = [feature,stage1,stage2,predict_layers_stage1,predict_layers_stage2]

    return net_dict, concats

def block_cell(tensor_in, net_block):
    t=tensor_in
    for i, cell in enumerate(net_block):
        assert(type(cell) is dict)
        nn_name=cell.keys()[0]
        nn_ksize=cell[nn_name]
        with tf.variable_scope(nn_name) as scope:
            kernel=tf.get_variable('w', shape=nn_ksize)
            bias=tf.get_variable('b',shape=nn_ksize[-1])
            if 'conv' in nn_name:
                t=tf.nn.conv2d(t, kernel, strides=[1,1,1,1], padding='SAME')
                t=tf.nn.bias_add(t, bias)
                if i<len(net_block)-1:
                    t=tf.nn.relu(t)
            elif 'pool' in nn_name:
                t=tf.nn.max_pool(t, nn_ksize, nn_ksize, padding='VALID')
    return t
    
def make_cpm(tensor_in):
    net_dict, concats=make_net_dict()
    net_layers=net_dict[:-2]
    pred_layers=net_dict[-2:]
    
    pred_layer_names=[l.keys()[0] for l in pred_layers]

    t=tensor_in
    tensor_out=[]
    tensor_to_concat=[]
    for block in net_layers:
        block_layer_names=[l.keys()[0] for l in block]
        
        if len(tensor_to_concat)==2:
            t=tf.concat(tensor_to_concat, axis=1)
            tensor_to_concat=[]

        t=block_cell(t, block)
        
        for name in block_layer_names:
            if name in pred_layer_names:
                tensor_out.append(t)
            for concat in concats:
                if name in concat[0:2]:
                    tensor_to_concat.append(t)
        
    return tensor_out