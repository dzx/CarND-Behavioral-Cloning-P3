#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 01:02:22 2017

@author: dzx
"""

import csv
import cv2 as cv
from moviepy.editor import ImageSequenceClip
import numpy as np
import random
import matplotlib.pyplot as plt
       
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D
from sklearn.model_selection import train_test_split
import sklearn


lines = []
images = []
measurements = []

FLIP_IMAGES = True
USE_SIDE_VIEW = True
SIDE_CORRECTION = .20
EPOCHS = 1
DROPOUT_RATE =.4
ZERO_ANGLE_RATE = .3
BATCH_SIZE = 24

#_show_image = True

def process_line(line, flip_image=False, path_index=0, correction=0):
    img_path = line[path_index]
    image = np.flip(cv.imread(img_path), 2)
#    global _show_image 
#    if _show_image and random.random() < .01 :
#        _show_image = False
#        plt.imshow(image)
#        plt.show()
    measurement = float(line[3])
    if(measurement or (random.random() < ZERO_ANGLE_RATE)):
        images.append(image)
        measurements.append(measurement + correction)
        if flip_image: # and measurement :
            image_flipped = np.fliplr(image)
            images.append(image_flipped)
            measurements.append(-measurement)
            
def generator(samples, batch_size=32):
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
            if (measurement or (random.random() < ZERO_ANGLE_RATE)):
                if(random.random() < .5):
                    measurement += SIDE_CORRECTION
                    img_path = batch_sample[1]
                else:
                    measurement -= SIDE_CORRECTION
                    img_path = batch_sample[2]
            image = np.flip(cv.imread(img_path), 2)
            batch_imgs.append(image)
            batch_angles.append(measurement)
            if FLIP_IMAGES: # and measurement :
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
    # Convolutional. Output = 10x10x16.['training set', 'validation set'], loc='upper right'
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

with open("./recordings/driving_logs2.csv") as logFile:
    reader = csv.reader(logFile)
    [lines.append(line) for line in reader]

#for i, line in enumerate(lines):
##    measurements.append(float(line[3]))
#    
#    if not( i % 1000):
#        print("Loading image {} of {}".format(i, len(lines)))['training set', 'validation set'], loc='upper right'
#    process_line(line, FLIP_IMAGES)
#    if USE_SIDE_VIEW :
#        process_line(line, FLIP_IMAGES, 1, SIDE_CORRECTION)
#        process_line(line, FLIP_IMAGES, 2, -SIDE_CORRECTION)

#clip = ImageSequenceClip(images, fps=25)
#clip.write_videofile("training.mp4")


#X_train = np.array(images)
#del images
#y_train = np.array(measurements)
#del measurements
#
#fig, ax = plt.subplots()
#ax.hist(y_train, bins=30)
#plt.show()

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)
epoch_sample = len(train_samples)
epoch_val = len(validation_samples)
if FLIP_IMAGES:
    epoch_sample *= 2
    epoch_val *= 2

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)) , input_shape=(160, 320, 3)))
model.add(Lambda(lambda x : x/127.5 - 1.))
model = create_nvnet(model)
#model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
#model.fit(X_train, y_train, batch_size=64, nb_epoch=EPOCHS, validation_split=.2, shuffle=True)
history_obj = model.fit_generator(train_generator, samples_per_epoch=epoch_sample, validation_data=validation_generator,
                    nb_val_samples= epoch_val, nb_epoch=EPOCHS)
model.save("model.h5")
print("Model saved")
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

