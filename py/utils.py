#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:41:20 2019

@author: mingrui
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

def show_augment(n_rows, n_cols):
    root = '/home/mingrui/udacity/self-driving/CarND-Behavioral-Cloning-P3/'
    img0 = mpimg.imread(os.path.join(root, 'augmentation_examples/original.jpg'))
    img1 = mpimg.imread(os.path.join(root, 'augmentation_examples/shearing.jpg'))
    img2 = mpimg.imread(os.path.join(root, 'augmentation_examples/brightness_change.jpg'))
    img3 = mpimg.imread(os.path.join(root, 'augmentation_examples/shadow.jpg'))
    img4 = mpimg.imread(os.path.join(root, 'augmentation_examples/translation.jpg'))
    img5 = mpimg.imread(os.path.join(root, 'augmentation_examples/flip.jpg'))
    
    imgs = [img0, img1, img2, img3, img4, img5]
    titles = ['original', 'shearing', 'brightness_change', 'shadow', 'translation', 'flip']
    count = 1
    
    img_cols = 3
    img_rows = 2
    
    hsize = img_cols * n_cols
    vsize = img_rows * n_rows
    fig = plt.figure(figsize=(hsize,vsize))
    
    for r in range(n_rows):
        for c in range(n_cols):
            img = imgs[count-1]
            
            sub = fig.add_subplot(n_rows, n_cols, count)
            sub.set_aspect('auto')
        
            sub.imshow(img)
            sub.set_title(titles[count-1])
        
            count += 1
            sub.axis('off')
            
            
    plt.show()

if __name__ == '__main__':
    show_augment(3, 2)