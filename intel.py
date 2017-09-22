#et ta -*- coding: utf-8 -*-
"""
Created on Sun May  7 07:37:54 2017

@author: ksevta
"""
#from functions import Vgg11,Vgg19,save_file
import numpy as np
from keras.optimizers import SGD
from keras.optimizers import adamax
import keras
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K	
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Conv2D,Convolution2D,MaxPooling2D
from keras.layers import Dropout,Flatten,Dense,Activation
from keras.regularizers import l2

import pandas as pd
import glob
    
def save_file(width,height,depth):
    
    path = './data/train/**/*.jpg'
    img_data = []
    i = 0
    Type = []
    print('[info] loading images...')
    for filename in glob.glob(path):
        img = cv2.imread(filename)   
        if img.size != None:
            if depth == 1:                
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img,(width,height), interpolation = cv2.INTER_LINEAR)
            #img = img.flatten()            
            img_data.append(img)
            s = 'Type'+(list((filename.split('/'))[3]))[5]
            Type.append(s)
            i += 1
    file1 = './data store/data'+str(width)+'x'+str(height)+'x'+str(depth) 
    file2 = './data store/labels'+str(width)+'x'+str(height)+'x'+str(depth) 
    np.save(file1,img_data) 
    np.save(file2,Type)  
    return 
    
def vgg11(width,height,depth,classes,weights = None):
    model = Sequential()
    # first set of CONV => RELU => POOL
    model.add(Conv2D(20, 5, 5, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.001)))

    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(50, 5, 5, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500,kernel_regularizer=l2(0.001)))
    model.add(Activation("tanh"))
    model.add(Dropout(0.1))

    # softmax classifier
    model.add(Dense(classes,kernel_regularizer=l2(0.001)))
    model.add(Activation("softmax"))
	
    return model
	
def vgg19(width,height,depth,classes,weights = None):
        
	model = Sequential()

	model.add(Conv2D(64, 3, 3, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.001)))
	model.add(Activation("relu"))
	model.add(Conv2D(64, 3, 3, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.001)))
	model.add(Activation("relu"))

	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))
	# second set of CONV => RELU => POOL
	model.add(Conv2D(128, 3, 3, border_mode="same"))
	model.add(Activation("relu"))
	model.add(Conv2D(128, 3, 3, border_mode="same"))
	model.add(Activation("relu"))

	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.4))

	model.add(Conv2D(256, 3, 3, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.0005)))
	model.add(Activation("relu"))
	model.add(Conv2D(256, 3, 3, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.001)))
	model.add(Activation("relu"))
	model.add(Conv2D(256, 1, 1, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.001)))
	model.add(Activation("relu"))

	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(512, 3, 3, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.001)))
	model.add(Activation("relu"))
	model.add(Conv2D(512, 3, 3, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.001)))
	model.add(Activation("relu"))
	model.add(Conv2D(512, 1, 1, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.001)))
	model.add(Activation("relu"))

	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.1))

	model.add(Conv2D(512, 3, 3, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.001)))
	model.add(Activation("relu"))
	model.add(Conv2D(512, 3, 3, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.001)))
	model.add(Activation("relu"))
	model.add(Conv2D(512, 1, 1, border_mode="same",input_shape=(height, width,depth),kernel_regularizer= l2(0.001)))
	model.add(Activation("relu"))	

	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	# set of FC => RELU layers


	model.add(Flatten())
	model.add(Dense(1044,kernel_regularizer=l2(0.001)))
	model.add(Activation("relu"))

	model.add(Dense(1044,kernel_regularizer=l2(0.001)))
	model.add(Activation("relu"))

	# softmax classifier
	model.add(Dense(classes,kernel_regularizer=l2(0.001)))
	model.add(Activation("softmax"))

	return model


save_file(64,64,3)
ap = argparse.ArgumentParser()

ap.add_argument("-w","--weights",type = int,help = "(optional) load weight or not")
#ap.add_argument("-l","--load-model",type = int,help = "(optional) load weight or not")
ap.add_argument("-s","--save-model",type = int,help = "(optional) save model or not")
ap.add_argument("-wi","--width",type = int,help = "(optional) width of image")
ap.add_argument("-he","--height",type = int,help = "(optional) height of image")
ap.add_argument("-de","--depth",type = int,help = "(optional) depth of image")
args = vars(ap.parse_args())

''' convert image to array '''

''' load data '''   

#save_file(32,32,3)

if args['width'] == None or args['height'] == None or args['depth'] == None:
	width = 32
	height = 32
	depth = 3
else:
	width = args['width']
	height = args['height']
	depth = args['depth']


''' data loading '''

data =  np.load('./data store/data' + str(width) + 'x' + str(height) + 'x' + str(depth)+'.npy')
labels = np.load('./data store/labels' + str(width) + 'x' + str(height) + 'x' + str(depth)+'.npy')
le = LabelEncoder()
labels = le.fit_transform(labels)


data = data.astype('float32')
data = data/255

trainData,testData,trainLabels,testLabels = train_test_split(data,labels,test_size=0.4, random_state=17)

labels = np_utils.to_categorical(labels)
#trainData= trainData[:, :, :,np.newaxis]
#testData= testData[:,:, :,np.newaxis]

#trainLabels = np_utils.to_categorical(trainLabels, 3)
#testLabels = np_utils.to_categorical(testLabels, 3)


datagen = ImageDataGenerator(width_shift_range =0.05,height_shift_range=0.05, rotation_range=0.3, zoom_range=0.3)
datagen.fit(data)

																																																																																																																																																																																																																																																																																								
model = vgg11(32,32,3,3,args["weights"])

model.compile(optimizer='adamax',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print('[info] fitting generator')
model.fit_generator(datagen.flow(trainData,trainLabels, batch_size=1, shuffle=True), nb_epoch=200, samples_per_epoch=len(trainData), 
	verbose=2, validation_data=(testData, testLabels))
#print(model_.history())

print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(testData, testLabels,batch_size=15, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

if args["save_model"] > 0:
    model.save_weights('output/weights32x32', overwrite=True)
