# **Behavioral Cloning** 


## Behavioral Cloning Project

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Files Submitted

Submission includes all required files and can be used to run the simulator in autonomous mode

* drive_net.py containing architecture of the neural network model.
* data_generator.py for data loading and augmentation.
* experiment_pipeline.py containing the overall experiment training and model saving pipeline, as well as the parameter class containing all the parameteres needed in the experiment.
* drive.py for driving the car in autonomous mode.
* model.h5 containing a trained convolution neural network.
* writeup_report.md summarizing the results.
* track1.mp4 and track2.mp4 recording of my vehicle driving autonomously on the first and the second track respetively.

## Usage of the Code

Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing:

```
python drive.py model.h5
```

And to train the model for driving, we can run:
```
python experiment_pipeline.py
```

## Experiment Process

My experiment process for this project can be divided into two stages:

### 1. Working on the first track, with data collected from the first track.

By following the project instuctions, I first drived the car in the training mode on the first track. And 3981 data samples were recorded. Then I trained a model with these data. The resulting vehicle ran out of the way or ran into the water. Then as suggested by the project instructions, I augmented the training data. In addition to the flipping described in the instructions, I also added shearing and transliation transformations. After training the model on the augmented data, the vehicle could move very well on the first track.

### 2. Working on both tracks, with data collected from both tracks

I tried the model obtained above on the seond track directly. It failed at the very beginning: the car turned left and hit the left boundary directly. So I simply collected the data from the second track, which was also suggested by the project instructions, by driving in the training mode on the second track. 2043 data samples were recorded. Then combined with the data collected from the first track, I trained another model based on the 6024 data samples collected from both tracks,together with the data augmentaions mentioned above. This new model could still work well on the first track, but failed in the middle of the second track, where there were some shadows on the road. So I augmented the data by changing the brighness of the images. In addition, I also added horizontal shadows. Then I re-trained the model on the augmented data. This final model worked very well on both of the tracks.

## Model Training

### 1. Model architecture

The model built for this project is defined in the drive_net.py. It contains three parts:

* Preprocess layers. These layers perform image normalization, image croping as suggested in the project material, and a 1*1 convolutional layer. A rate of 20% **Drop out** is applied to overcome overfitting.
* Convolutional layers. These are just normal convolutional layers with 3*3 kernels. And **max poolint** and **batch normalization** are applied in each convolutional year. According to the experience in the previoius project of **German traffic sign classification**, batch normalization is helpful to stablize the training process when data augmentation is applied in each batch.
* Fully connected layers. There are two fully connected layers for calulating the final steering angle. **Batch normalization** is also applied. And **drop out** is applied here to overcome overfitting. 
The model used an adam optimizer, so the learning rate was not tuned manually.

Details of the model architecture are as the following:

Layer (type)                 Output Shape              Param #   

=================================================================

lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 300, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 90, 300, 8)        32        
_________________________________________________________________
batch_normalization_1 (Batch (None, 90, 300, 8)        32        
_________________________________________________________________
activation_1 (Activation)    (None, 90, 300, 8)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 90, 300, 8)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 88, 298, 16)       1152      
_________________________________________________________________
batch_normalization_2 (Batch (None, 88, 298, 16)       64        
_________________________________________________________________
activation_2 (Activation)    (None, 88, 298, 16)       0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 44, 149, 16)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 42, 147, 32)       4608      
_________________________________________________________________
batch_normalization_3 (Batch (None, 42, 147, 32)       128       
_________________________________________________________________
activation_3 (Activation)    (None, 42, 147, 32)       0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 21, 73, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 19, 71, 64)        18432     
_________________________________________________________________
batch_normalization_4 (Batch (None, 19, 71, 64)        256       
_________________________________________________________________
activation_4 (Activation)    (None, 19, 71, 64)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 9, 35, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 20160)             0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 20160)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               2580480   
_________________________________________________________________
batch_normalization_5 (Batch (None, 128)               512       
_________________________________________________________________
activation_5 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               16384     
_________________________________________________________________
batch_normalization_6 (Batch (None, 128)               512       
_________________________________________________________________
activation_6 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 129       

=================================================================

Total params: 2,622,721
Trainable params: 2,621,969
Non-trainable params: 752

### 2. Training and Parameters

All the data are divided into two parts: training and validation. For each original image colleted, 3 transformed images are generated in each batch. The detials of the augmentaion will be desribed later. All the parameteres are compiled together in the ExperimentParam class, which is defined the experiment_pipeline.py file.

## Data Augmentation

Data augmentaion is the key part that can make model work.Here are some details:

1. New images are generated in each batch during training. This is done in the data_generator.py, which defines the data generator.

2. In order to make the traning stable, **batch normalization** layers are applied in my network, defined in drive_net.py. 

3. For every generated new image, in addition to the corresponing image transformation, the **label**, i.e. the steering angle, also needs to be calculated. 

4. The following agumentations are generated:

    * As suggested by the instructions of the project, center, left and right images are all added in the training data. And for the left/right images, a correction of 0.35 is added and substracted from the original steering repectively.
    * Image shearing: defined in DataGenerator.shearing. Inspired by adding left and right images into the training data, each image is sheared horizontally. The resulting steering equals to
    ```sheared_steering = original_steering + atan2(delta, n_rows/2)```, where `n_rows` is number of rows of each image, and `delta` is a random integer ,either positive or negative, indicating how many pixels the center of the image should move horizontally. This can generate many more steering angles than the original input images. And it can generate the steering angles for both left turns and right turns.
    * Flipping: defined in DataGenerator.flip. Each image is flipped with a probablility of 50%. This is to reduce the bias of the steering angle. For example in the first track most of time we steer left of go straight. There is very few cases where we need to steer right. Flipping can give more right steering data. The resulting steering is just the negative value of the original steering angle.
    * Translation: defined in DataGenerator.translation respectively. The resulting steering angle is the same as the original one.
    * Brightness change: defined in DataGenerator.change_brightness. Orignally I would do something like histogram equalization to reduce the impact of variance of the brightness in the images. However, it seems that there is no such layer in the kerras, and it is not easy to do image pre-process separately in the current pipeline setup. So images with different brightness are also generated. An example is shown in Figure.
    * Adding shadows:  Defined in DataGenerator.add_horizontal_shadow This is mainly to deal with the shadows in the second track, where there are several places that contain shadows on the road.  I also wrote the code for adding vertical shadows. But it turned out that only adding horizongtal shadows could already make model work well.  

An example of augmented images is shown below. The augmentations are applied sequentially: original-->shearing-->bightness_change-->shadow-->translation-->flip.

![alt text][image1]

Note that the translation amount is within 3 pixels, so the difference between translation and its input shadow image is very small.


[//]: # (Image References)
[image1]: ./augmentation_examples/augmentation.png


## Results

The model with the above described architecture is trained with the augmented data described above can drive automously very well on both tracks. The video recordings on both tracks, track1.mp4 and track2.mp4, are included in the repository.
