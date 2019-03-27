#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:48:38 2019

@author: mingrui
"""

from drive_net import drive_net
from data_generator import DataGenerator

class ExperimentParam:
    def __init__(self):
        # input image size
        self._n_rows = 160
        self._n_cols = 320
        self._n_channels = 3
        
        # data generation and augmentation
        self._validation_ratio = 0.2
        self._data_path =  '/home/mingrui/udacity/self-driving/CarND-Behavioral-Cloning-P3/data1/'
        self._n_transforms = 3
        
        # image processing
        self._top_crop = 50
        self._bottom_crop = 20
        self._left_crop = 10
        self._right_crop = 10
        
        # learning parameters
        self._learning_rate = 0.001
        self._n_epochs = 20
        self._batch_size = 32
        
        

def train(param):
    # data generator
    dg = DataGenerator(param)
       
    # model graph
    model = drive_net(param)
    
    # training
    model.fit_generator(dg._train_generator, 
            steps_per_epoch=dg._train_steps, 
            validation_data=dg._validation_generator, 
            validation_steps=dg._valid_steps, 
            epochs=param._n_epochs, verbose=1)

    # save the model
    model.save('model.h5')
    


if __name__ == '__main__':
    param = ExperimentParam()
    train(param)