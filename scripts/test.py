#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:59:36 2020

@author: manasd28
"""

import cv2 
import keras
import numpy as np

img = cv2.imread('../test_data/images/download.png')

model = keras.models.load_model('../data_files/model-13x13.h5')

model_labels = ['Speed limit (20km/h)',
     'Speed limit (30km/h)',
     'Speed limit (50km/h)',
     'Speed limit (60km/h)',
     'Speed limit (70km/h)',
     'Speed limit (80km/h)',
     'End of speed limit (80km/h)',
     'Speed limit (100km/h)',
     'Speed limit (120km/h)', 'No passing',
     'No passing for vehicles over 3.5 metric tons',
     'Right-of-way at the next intersection',
     'Priority road',
     'Yield',
     'Stop',
     'No vehicles',
     'Vehicles over 3.5 metric tons prohibited',
     'No entry',
     'General caution',
     'Dangerous curve to the left',
     'Dangerous curve to the right',
     'Double curve',
     'Bumpy road',
     'Slippery road',
     'Road narrows on the right',
     'Road work',
     'Traffic signals',
     'Pedestrians',
     'Children crossing',
     'Bicycles crossing', 'Beware of ice/snow',
     'Wild animals crossing',
     'End of all speed and passing limits',
     'Turn right ahead',
     'Turn left ahead',
     'Ahead only',
     'Go straight or right',
     'Go straight or left',
     'Keep right',
     'Keep left',
     'Roundabout mandatory',
     'End of no passing',
     'End of no passing by vehicles over 3.5 metric tons']


img 
pred = model.predict(img)
print(np.argmax(model_labels[pred]))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pred = model.predict(img)
print(np.argmax(model_labels[pred]))