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
import cv2
import math

def gamma_correction(img, gamma=1.0):
    table = np.array([((i/255.0)**gamma)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)

class DataGenerator:
    def __init__(self, param):
        self._param = param
        
        csv_fname = os.path.join(param._data_path, 'driving_log.csv')
        self._df = pd.read_csv(csv_fname)
        
        indices = list(range(len(self._df)))
        np.random.shuffle(indices)
        
        # indices for training and validation
        num_data = len(indices)
        num_train = int(num_data * (1 - param._validation_ratio))
        num_valid = num_data - num_train
        self._train_indices = indices[:num_train]
        self._validation_indices = indices[num_train:]
        
        batch_size = param._batch_size
        self._train_steps = num_train // batch_size
        self._valid_steps = num_valid // batch_size
        
        self._train_generator = self.generator(batch_size, self._train_indices, is_training=True)
        self._validation_generator = self.generator(batch_size, self._validation_indices)
        
    # data generator
    def generator(self, batch_size, data_indices, is_training=False):
        num_data = len(data_indices)
        
        while True:
            np.random.shuffle(data_indices)
            for offset in range(0, num_data, batch_size):
                # indices of current batch
                batch_indices = data_indices[offset : offset + batch_size]
                
                batch_images = []
                batch_steering = []
                
                for idx in batch_indices:
                    for name in ['center', 'left', 'right']:
                        img = mpimg.imread(os.path.join(self._param._data_path, (self._df[name][idx].strip())))
                        batch_images.append(img)
                    
                    steering = self._df['steering'][idx]
                    batch_steering.append(steering)
                    batch_steering.append(steering + 0.35)
                    batch_steering.append(steering - 0.35)
                    
                # data autmentation
                if is_training:
                    N = len(batch_images)
                    for i in range(N):
                        for t in range(self._param._n_transforms):
                            transformed_img, transformed_steering = self.data_transform(batch_images[i], batch_steering[i])
                            batch_images.append(transformed_img)
                            batch_steering.append(transformed_steering)
                            
                X = np.array(batch_images)
                y = np.array(batch_steering)
                

                yield X, y
    
    def load_one_img(self):
        idx = np.random.randint(0, len(self._df)-1)
        img = mpimg.imread(os.path.join(self._param._data_path, (self._df['center'][idx].strip())))
        return img
        
    
    def data_transform(self, img, steering):
         # shearing
        transformed_img, transformed_steering = self.shearing(img, steering)
        
        # brightness change
        transformed_img = self.change_brightness(transformed_img)
        
        # shadow
        if np.random.random() < self._param._shadow_rate:
            if np.random.random() < 0.5:
                transformed_img = self.add_horizontal_shadow(transformed_img)
            else:
                transformed_img = self.add_vertical_shadow(transformed_img)
        
        # scaling and translation
        transformed_img = self.scaling(transformed_img)
        transformed_img = self.translation(transformed_img)
        
        # flip
        if np.random.random() < self._param._flip_rate:
           transformed_img, transformed_steering = self.flip(transformed_img, transformed_steering)
           
        return transformed_img, transformed_steering
    
     
    # shearing
    def shearing(self, img, steering):
        n_rows, n_cols = img.shape[:2]
            
        # sheering
        center_x = n_rows // 2
        center_y = n_cols // 2
        
        src = np.array([[center_x, center_y], [center_x+5, center_y], [n_rows-1, center_x]], dtype=np.float32)
        dst = np.copy(src)
        delta = np.random.randint(self._param._shear_range[0], self._param._shear_range[1])
        dst[0] += [delta, 0]
        dst[1] += [delta, 0]
            
        shear_m = cv2.getAffineTransform(src, dst)
        transformed_img  = cv2.warpAffine(img, shear_m, (n_cols, n_rows))
        transfomred_steering = steering + math.atan2(delta, center_y)
        
        return transformed_img, transfomred_steering

    # brightness change
    def change_brightness(self, img):
        # brightness change
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v_factor = np.random.uniform(self._param._brightness_range[0], self._param._brightness_range[1])
        hsv[:, :, 2] = hsv[:, :, 2] * v_factor
        transformed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return transformed_img
    
    # horzontal shadow
    def add_horizontal_shadow(self, img):
        transformed_img = np.copy(img)
        row = np.random.randint(self._param._horizontal_shadow_range[0], self._param._horizontal_shadow_range[1])
        hsv = cv2.cvtColor(img[:row, :, :], cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * self._param._shadow_factor
        transformed_img[:row, :, :] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
        return transformed_img
    
    # vetical shadow
    def add_vertical_shadow(self, img):
        n_cols = img.shape[1]
        transformed_img = np.copy(img)
        
        # starting and ending columns of the vertical shadow
        col = np.random.randint(self._param._vertical_shadow_range[0], self._param._vertical_shadow_range[1])
        if col < n_cols/2:
            start = col
            end = n_cols
        else:
            start = 0
            end = col
        hsv = cv2.cvtColor(img[:, start:end, :], cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * self._param._shadow_factor
        transformed_img[:, start:end, :] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return transformed_img
    
    # scaling
    def scaling(self, img):
        n_rows, n_cols = img.shape[:2]
        scale = np.random.uniform(self._param._scaling_range[0], self._param._scaling_range[1])
        rot_m = cv2.getRotationMatrix2D((n_cols/2, n_rows/2), 0, scale)
        o = cv2.warpAffine(img, rot_m, (n_cols, n_rows))
        return o
    
    # translation
    def translation(self, img):
        n_rows, n_cols = img.shape[:2]
        
        tx, ty = np.random.uniform(-self._param._translation_range, self._param._translation_range, 2)
        trans_m = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        transformed_img = cv2.warpAffine(img, trans_m, (n_cols, n_rows))
        
        return transformed_img
    
    
    def flip(self, img, steering):
        # flip
        transformed_img = cv2.flip(img, flipCode=1)
        transformed_steering = -steering
        
        return transformed_img, transformed_steering