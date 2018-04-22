# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:57:45 2018

@author: JYF
"""

import numpy as np
import tensorflow as tf

class ModelLoader:
    def __init__(self, load_mode='ckpt',model_path='',skip=None):
        self.model_type_=load_mode
        if self.model_type_=='ckpt':
            pass
        elif self.model_type_=='npy':
            pass
        else:
            raise Exception('Unknown model type')
        self.model_path_=model_path
        self.skipped_layers=skip
    
    def load_model(self, session=None):
        if session is None:
            raise Exception('No valid session found')
        if self.model_type_=='ckpt':
            self.load_model_ckpt(session=session)
        elif self.model_type_=='npy':
            self.load_model_npy(session=session)
        print('Pretrained model loaded')
        
    def load_model_npy(self,session=None):
        w_dict=np.load(self.model_path_, encoding='bytes').item()
        for name in w_dict:
            if name not in self.skipped_layers:
                with tf.variable_scope(name, reuse=True):
                    for param in w_dict[name]:
                        if len(param.shape)==1:
                            session.run(tf.get_variable('b',trainable=True).assign(param))  #run tensor
                        else:
                            session.run(tf.get_variable('w', trainable=True).assign(param))
    
    def load_model_ckpt(self, session=None):
        saver=tf.train.Saver()
        saver.restore(session, self.model_path_)