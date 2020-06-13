#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:09:15 2020

@author: linshihuan
"""
# In[1]

import csv
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


    
def get_data(filename):
    

    with open(filename) as training_file:
        csv_file = csv.reader(training_file, delimiter=',')
        images = []
        labels = []
        for list in csv_file:
            images_2d = [list[i:i+28] for i in range(1, 785, 28)]
            images.append(images_2d)
            labels = np.append(labels, list[0])
        labels = np.delete(labels,0,0)
        labels = labels.astype(int)
        images = np.array(images)
        images = np.delete(images,0,0)
        images = images.astype(float)
    
        
    return images, labels

#If you want to try txt processing 
'''
    with open(filename) as training_file:
        txt_file = np.loadtxt(training_file, delimiter=',', skiprows=1)
        images = []
        labels = []
        
        for list in txt_file:
            images_2d = [list[i:i+28] for i in range(1, 785, 28)]
            images.append(images_2d)
            labels = np.append(labels, list[0])
        training_file.close()
        labels = labels.astype(int)
        images = np.array(images)
        images = images.astype(float)
        images_2d = None
        txt_file = None
        
        training_file.close()
    return images, labels
'''
path_file= os.path.abspath('.')
path_sign_mnist_train = path_file + '/sign_mnist_train.csv'
path_sign_mnist_test = path_file + '/sign_mnist_test.csv'
testing_images, testing_labels = get_data(path_sign_mnist_test)
training_images, training_labels = get_data(path_sign_mnist_train)
        

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)



# In[2]
training_images = np.expand_dims(training_images, axis=3)# Your Code Here
testing_images = np.expand_dims(testing_images, axis=3)# Your Code Here

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=0.2,
    shear_range=0,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


validation_datagen = ImageDataGenerator(rescale=1./255)

    
# Keep These
print(training_images.shape)
print(testing_images.shape)

# In[3]

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(25, activation='softmax')])

# Compile Model. 
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam' , metrics=['accuracy'])

# Train the Model
history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=25),
                              steps_per_epoch=len(training_images)/25,
                              epochs=5,
                              validation_data = validation_datagen.flow(testing_images, testing_labels, batch_size=25),
                              validation_steps=len(testing_images)/25)

model.evaluate(testing_images, testing_labels)

# In[4]


import matplotlib.pyplot as plt
acc = history.history['accuracy']# Your Code Here
val_acc = history.history['val_accuracy']# Your Code Here
loss = history.history['loss']# Your Code Here
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()