import itertools
import os
import time

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB7  as PretrainedModel, preprocess_input
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from glob import glob

#data loading
folders = glob('C:/Users/Dell/Desktop/archive/lung_colon_image_set' + '/*')

lung_images = glob('C:/Users/Dell/Desktop/archive/lung_colon_image_set/lung_image_sets' + '/*/*.jpeg')

lung_data_directory = 'C:/Users/Dell/Desktop/archive/lung_colon_image_set/lung_image_sets/'

#creating a data generator for the lung images
lung_data_generator = ImageDataGenerator(rescale=1./255, validation_split = 0.2)
#splitting into training and validation set,setting parameters
lung_training = lung_data_generator.flow_from_directory(lung_data_directory,
                                    class_mode = "categorical",
                                    target_size = (64, 64),
                                    color_mode="rgb",
                                    batch_size = 64, 
                                    shuffle = True,
                                    subset='training',
                                    seed = 420)

lung_validation = lung_data_generator.flow_from_directory(lung_data_directory,
                                      class_mode = "categorical",
                                      target_size = (64, 64),
                                      color_mode="rgb",
                                      batch_size = 64, 
                                      shuffle = True,
                                      subset='validation',
                                      seed = 69)




colon_images = glob('C:/Users/Dell/Desktop/archive/lung_colon_image_set/colon_image_sets' + '/*/*.jpeg')

colon_data_directory = 'C:/Users/Dell/Desktop/archive/lung_colon_image_set/colon_image_sets/'

colon_data = ImageDataGenerator(rescale=1./255, validation_split = 0.2)

colon_training = colon_data.flow_from_directory(colon_data_directory,
                                    class_mode = "categorical",
                                    target_size = (64, 64),
                                    color_mode="rgb",
                                    batch_size = 64, 
                                    shuffle = True,
                                    subset='training',
                                    seed = 420)

colon_validation = colon_data.flow_from_directory(colon_data_directory,
                                      class_mode = "categorical",
                                      target_size = (64, 64),
                                      color_mode="rgb",
                                      batch_size = 64, 
                                      shuffle = True,
                                      subset='validation',
                                      seed = 69)





# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax')) # Set `num_classes` based on the number of classes in your dataset

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(
        lung_training,
        epochs=20,
        validation_data=lung_validation,
        )

# Plot the accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()




history=model.fit(
        colon_training,
        epochs=20,
        validation_data=colon_validation,
        )


