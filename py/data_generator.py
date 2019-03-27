#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:47:56 2019

@author: mingrui
"""

import pandas as pd
import numpy as np
import os
import matplotlib.image as mpimg

class DataGenerator:
    def __init__(self, data_path, validation_ratio, batch_size):
        csv_fname = os.path.join(data_path, 'driving_log.csv')
        self._data_path = data_path
        self._df = pd.read_csv(csv_fname)
        
        indices = list(range(len(self._df)))
        np.random.shuffle(indices)
        
        # indices for training and validation
        num_data = len(indices)
        num_train = int(num_data * (1 - validation_ratio))
        num_valid = num_data - num_train
        self._train_indices = indices[:num_train]
        self._validation_indices = indices[num_train:]
        
        self._batch_size = batch_size
        self._train_steps = num_train // batch_size
        self._valid_steps = num_valid // batch_size
        
        self._train_generator = self.generator(self._batch_size, self._train_indices)
        self._validation_generator = self.generator(self._batch_size, self._validation_indices)
        
    # data generator
    def generator(self, batch_size, data_indices):
        num_data = len(data_indices)
        
        while True:
            np.random.shuffle(data_indices)
            for offset in range(0, num_data, batch_size):
                # indices of current batch
                batch_indices = data_indices[offset : offset + batch_size]
                
                batch_images = []
                batch_angles = []
                
                for idx in batch_indices:
                    for name in ['center', 'left', 'right']:
                        img = mpimg.imread(os.path.join(self._data_path, (self._df[name][idx].strip())))
                        batch_images.append(img)
                    
                    angle = self._df['steering'][idx]
                    batch_angles.append(angle)
                    batch_angles.append(angle + 0.35)
                    batch_angles.append(angle - 0.35)
                    
                X = np.array(batch_images)
                y = np.array(batch_angles)

                yield X, y
                    