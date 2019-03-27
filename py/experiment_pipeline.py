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
        self._n_rows = 160
        self._n_cols = 320
        self._n_channels = 3
        
        self._learning_rate = 0.001
        

def train(param):
    # data generator
    data_path = '/home/mingrui/udacity/self-driving/CarND-Behavioral-Cloning-P3/data1/'
    dg = DataGenerator(data_path, validation_ratio=0.2, batch_size=128)
       
    # model graph
    model = drive_net(param)
    
    # training
    model.fit_generator(dg._train_generator, 
            steps_per_epoch=dg._train_steps, 
            validation_data=dg._validation_generator, 
            validation_steps=dg._valid_steps, 
            epochs=10, verbose=1)

    # save the model
    model.save('model.h5')
    


if __name__ == '__main__':
    param = ExperimentParam()
    train(param)