#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:55:25 2019

@author: mingrui
"""

#import tensorflow as tf
#from tensorflow.contrib.layers import flatten, conv2d, l2_regularizer, max_pool2d, fully_connected, dropout, batch_norm

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Cropping2D
from keras.optimizers import Adam

def drive_net(param):
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(param._n_rows, param._n_cols, param._n_channels)))
    model.add(Conv2D(5, (5, 5)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(5, (5, 5)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(5, (5, 5)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(5, (5, 5)))
    model.add(MaxPooling2D())
    
    model.add(Flatten(input_shape=(param._n_rows, param._n_cols, param._n_channels)))
    
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer=Adam(lr=param._learning_rate))
    return model
    
