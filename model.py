#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 01:02:22 2017

@author: dzx
"""

import csv
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
       
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D
from sklearn.model_selection import train_test_split
import sklearn


lines = []

FLIP_IMAGES = True
SIDE_CORRECTION = .2 # steering correction for samples from side cams
EPOCHS = 3
DROPOUT_RATE =.4
ZERO_ANGLE_RATE = .1 # retention rate for samples with small steering angle
BATCH_SIZE = 32

SHOW_TRAIN_IMAGE = False # control diagnostic check of image decoding

            
def generator(samples, batch_size=32):
    show_image = SHOW_TRAIN_IMAGE
    num_samples = len(samples)
    while True :
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
        
            batch_imgs = []
            batch_angles = []
            for batch_sample in batch_samples:
                measurement = float(batch_sample[3])
                img_path = batch_sample[0]
                if not (abs(measurement) > .015 or random.random() < ZERO_ANGLE_RATE):
                    # substituting 'centered wheel' image for image from side cam
                    if(random.random() < .5):
                        # 50-50 chance of picking left or right side cam
                        measurement += SIDE_CORRECTION
                        img_path = batch_sample[1]
                    else:
                        measurement -= SIDE_CORRECTION
                        img_path = batch_sample[2]
                #load and convert OpenCV BGR -> RGB
                image = np.flip(cv.imread(img_path), 2)
                if show_image:
                    #One-time display of decoded image
                    show_image = False
                    plt.imshow(image)
                    plt.show()
                batch_imgs.append(image)
                batch_angles.append(measurement)
                if FLIP_IMAGES: # add mirrored image to reduce steering bias
                    image_flipped = np.fliplr(image)
                    batch_imgs.append(image_flipped)
                    batch_angles.append(-measurement)
            
            X_train = np.array(batch_imgs)
            Y_train = np.array(batch_angles)
            
            yield sklearn.utils.shuffle(X_train, Y_train)
            
                    

def create_lenet(model):
    #Output = 28x28x6.
    model.add(Conv2D(6, 7, 13, activation='relu', subsample=(3, 11)))
    #Pooling. Input = 28x28x6. Output = 14x14x6.
    model.add(MaxPooling2D())
    # Convolutional. Output = 10x10x16.
    model.add(Conv2D(16, 5, 5, activation='relu'))
    #Pooling. Input = 10x10x16. Output = 5x5x16.
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(43, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    return model

def create_nvnet(model):
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(64, 3, 3,  activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(50))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(10))
    model.add(Dropout(DROPOUT_RATE))
    return model

# Load sample index
with open("./recordings/driving_logs2.csv") as logFile:
    reader = csv.reader(logFile)
    [lines.append(line) for line in reader]


train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)
epoch_sample = len(train_samples)
epoch_val = len(validation_samples)
if FLIP_IMAGES: # Double set size because we are adding mirrored images
    epoch_sample *= 2
    epoch_val *= 2


model = Sequential()
# Pre-processing layer
model.add(Cropping2D(cropping=((50, 20), (0, 0)) , input_shape=(160, 320, 3)))
model.add(Lambda(lambda x : x/127.5 - 1.))
# Add actual learning model
model = create_nvnet(model)
# Output layer
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history_obj = model.fit_generator(train_generator, samples_per_epoch=epoch_sample, validation_data=validation_generator,
                    nb_val_samples= epoch_val, nb_epoch=EPOCHS, verbose=1)
model.save("model.h5")
print("Model saved")
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

