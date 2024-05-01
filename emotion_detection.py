import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

DataDirectory = "train/"
Classes = ['0', '1', '2', '3', '4', '5', '6']   
img_size = 224
training_data = []

for category in Classes: # Reading all the Images
    path = os.path.join(DataDirectory, category)
    class_num = Classes.index(category) #Label
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass
         
random.shuffle(training_data)

x, y = [], []

for features, label in training_data:
    x.append(features)
    y.append(label)

    
x = np.array(x).reshape(-1, img_size, img_size, 3)

#normalize the data
x = x/255.0
y = np.array(y)

model = tf.keras.applications.MobileNetV2()
base_input = model.layers[0].input
base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output)

new_model = keras.Model(inputs = base_input, outputs = final_output)
new_model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics = ['accuracy'])
new_model.fit(x, y, epochs = 25)