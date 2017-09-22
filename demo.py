from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K
import numpy as np
import pandas as pd

#K.set_image_dim_ordering('th')
#K.set_floatx('float32')

import pandas as pd
import numpy as np
np.random.seed(17)


''' data loading '''
folder_path =  './data store/32x32/'
dataType1 = np.load(folder_path+'imgToArrayType_1.npy')
dataType2 = np.load(folder_path+'imgToArrayType_2.npy')
dataType3 = np.load(folder_path+'imgToArrayType_3.npy')

dataType1_ad = np.load(folder_path+'imgToArrayType_1_ad.npy')
dataType2_ad = np.load(folder_path+'imgToArrayType_2_ad.npy')
dataType3_ad = np.load(folder_path+'imgToArrayType_3_ad.npy')


label1 = np.zeros((len(dataType1),), dtype = int)
label2 = np.ones((len(dataType2),),dtype = int)
label3 = np.ones((len(dataType3),),dtype = int) 
label3 += 1

label1_ad = np.zeros((len(dataType1_ad),), dtype = int)
label2_ad = np.ones((len(dataType2_ad),),dtype = int)
label3_ad = np.ones((len(dataType3_ad),),dtype = int) 
label3_ad += 1



data = np.concatenate((dataType1,dataType1_ad,dataType2,dataType2_ad,dataType3,dataType3_ad),0)
labels = np.concatenate((label1,label1_ad,label2,label2_ad,label3,label3_ad),0)
data = data.astype('float32')
data = data/255


def create_model(opt_='adamax'):
    model = Sequential()
    model.add(Convolution2D(4, 3, 3, activation='relu', input_shape=(32, 32,3))) #use input_shape=(3, 64, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(8, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model

def cleanImages():
    datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)
    datagen.fit(data)
    return datagen

def fitAndPredict():
    print("cleaning images")
    datagen=cleanImages()
    print("images cleaned")
    K.set_image_data_format = 'channels_last'
    model = create_model()
    x_train,x_val_train,y_train,y_val_train = train_test_split(data,labels,test_size=0.4, random_state=17)
    print("fitting data")
    model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True), nb_epoch=200, samples_per_epoch=len(x_train), verbose=2, 
    	validation_data=(x_val_train, y_val_train))
    #print("data fitted in model")
    #test_data = np.load('test.npy')
    #test_id = np.load('test_id.npy')
    #print("creating predictions")
    #predictions = model.predict_proba(test_data)
    #print("predictions made")
    return #predictions

def createSub():
    pred=fitAndPredict()
    #print("creating submission file")
    #df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
    #df['image_name'] = test_id
    #df.to_csv('submission.csv', index=False)
    #print("submission created")


if __name__ == '__main__':
    
    createSub()
