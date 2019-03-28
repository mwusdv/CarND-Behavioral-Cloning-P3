#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:55:25 2019

@author: mingrui
"""

#import tensorflow as tf
#from tensorflow.contrib.layers import flatten, conv2d, l2_regularizer, max_pool2d, fully_connected, dropout, batch_norm

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Cropping2D, Lambda, Dropout
from keras.optimizers import Adam

import cv2
import numpy as np


def drive_net(param):
    model = Sequential()
    
    # preprocessing
    model.add(Lambda(lambda x: (x/255.0-0.5)*2, input_shape=(param._n_rows, param._n_cols, param._n_channels)))
    model.add(Cropping2D(cropping=((param._top_crop, param._bottom_crop), 
                                   (param._left_crop, param._right_crop)), 
                                   input_shape=(param._n_rows, param._n_cols, param._n_channels)))
    model.add(Conv2D(8, (1,1), activation='relu'))
    model.add(Dropout(0.2))
      
    # conv layers
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    
    # fc layers
    model.add(Flatten())
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer=Adam(lr=param._learning_rate))
    return model
    
