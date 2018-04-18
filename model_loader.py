# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:57:45 2018

@author: JYF
"""

import numpy as np
import tensorflow as tf

class ModelLoader:
    def __init__(self, load_mode='ckpt',model_path=''):
        self.model_type_=load_mode
        if self.model_type_=='ckpt':
            pass
        elif self.model_type_=='npy':
            pass
        else:
            raise Exception('Unknown model type')
        self.model_path_=model_path
    
    def load_model_npy(self):
        pass
    
    def load_model_ckpt(self, session=None):
        pass