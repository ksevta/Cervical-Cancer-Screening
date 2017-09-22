# -*- coding: utf-8 -*-
"""
Created on Sun May  7 23:38:24 2017

@author: ksevta
"""
from keras.models import Sequential
from keras.layers import Conv2D,Convolution2D,MaxPooling2D
from keras.layers import Dropout,Flatten,Dense,Activation
from keras.regularizers import l2
class LeNet:
    @staticmethod
    def build(width,height,depth,classes,weights = None):
        
	'''
	model = Sequential()
	model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th', input_shape=(32, 32))) #use input_shape=(3, 64, 64)
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
	model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
	model.add(Dropout(0.2))
	    
	model.add(Flatten())
	model.add(Dense(12, activation='tanh'))
	model.add(Dropout(0.1))
	model.add(Dense(3, activation='softmax'))
	return model	
	'''      
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
		
        # if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
#if weightsPath is not None:
            #model.load_weights(weightsPath)

		# return the constructed network architecture
        
        
'''        
==============================================================================
        model = Sequential();
         
        model.add(Conv2D(64,(3,3), activation = 'relu' , input_shape= (width,height,depth)))
        model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))
        # model.add(Dropout(0.25))
         
        model.add(Conv2D(128,(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2))) 
         
        model.add(Conv2D(256,(3,3),activation='relu'))
        model.add(Conv2D(256,(3,3),activation='relu'))
         
        model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))
         
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(Conv2D(512,(3,3),activation='relu'))
         
        model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))
        
        model.add(Conv2D(512,(3,3),activation='relu'))
        model.add(Conv2D(512,(3,3),activation='relu'))
        
        model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))
        
        model.add(Flatten())
        model.add(Dense(4096,activation='relu'))
        model.add(Dense(4096,activation='relu'))
        model.add(Dense(classes,activation='softmax'))
        
        if weights is not None:
            model.load_weights('weights.hdf5')
        
        return model
=============================================================================

'''
