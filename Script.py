"""
Created on Wed Jun  2 14:09:04 2021
@author: xenificity
"""
#import libraries
from keras.layers import MaxPool2D,Flatten,Dense,Dropout, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.preprocessing.image import *
from keras.utils import np_utils
import matplotlib.pylab as plt
import tensorflow as tf
from PIL import Image as im
import pandas as pd
import numpy as np
import cv2
import os

os.chdir("D:/AI test") #setting working directory
#taking file path and labesl
train = ImageDataGenerator(rescale=1/255)
val = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('training',target_size=(300,300),batch_size=30)
validation_dataset = val.flow_from_directory('validation',target_size=(300,300),batch_size=30)
testing_dataset = test.flow_from_directory('testing',target_size=(300,300))

#Image showing
train_labels= train_dataset.classes
classes = train_dataset.class_indices
img = train_dataset[0][0]
img = np.reshape(img,(300,300,3))
cv2.imshow("image_instance",img)
cv2.waitKey(0) # wait for ay key to exit window
cv2.destroyAllWindows() # close all windows

#Model-1
model = Sequential()
model.add(Conv2D(128,(3,3),activation='swish',input_shape=(300,300,3)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64,(3,3),activation='swish'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(32,(3,3),activation='swish'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(16,(3,3),activation='swish'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(264,activation='swish'))
model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
epochs=20
model.summary() 

#Model-2
transfered=InceptionV3(include_top=False,weights='imagenet',input_tensor=None,input_shape=(None,None,3),pooling='avg',classes=6)
model=Sequential()
model.add((InputLayer(None,None,3)))
model.add(transfered)
model.add(Dropout(0.1))
model.add(Dense(6,activation='softmax'))
transfered.trainable=False
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
epochs=100
model.summary()

#Model-3
transfered=EfficientNetB3(include_top=False,weights='imagenet',input_shape=(None,None,3),classes=6)
model=Sequential()
model.add((InputLayer(300,300)))
model.add(transfered)
model.add(Dropout(0.2))
model.add(Dense(6,activation='softmax'))
transfered.trainable=True
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
batch_size=10
epochs=10
model.summary()

#Fitting model
model_fit = model.fit(train_dataset,epochs=epochs,validation_data=validation_dataset,callbacks=[ModelCheckpoint(filepath='./weights.hdf5',monitor='val_acc',verbose=1,save_best_only=True)])

#Model Prediction
images = []
for img in os.listdir('Testing'):
    img = os.path.join('Testing', img)
    img = image.load_img(img, target_size=(300, 300))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)
classes = model.predict_classes(images, batch_size=10)
print(classes_test)
