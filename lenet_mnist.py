# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages

#THEANO_FLAGS=device=cuda0
from pyimagesearch.cnn.networks import LeNet
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2
import functions
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)

#functions.img_to_array('Type_2_ad')

print("[INFO] downloading data...")
''' data loading '''
folder_path =  './data store/224x224/'
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



#data = np.concatenate((dataType1,dataType2,dataType3),0)
#labels = np.concatenate((label1,label2,label3),0)

labels = np.concatenate((label1,label1_ad,label2,label2_ad,label3,label3_ad),0)
data = np.concatenate((dataType1,dataType1_ad,dataType2,dataType2_ad,dataType3,dataType3_ad),0)

#print('initial shape ',labels.shape)
#data = data[1:500,:,:,:]
#labels = labels[1:500]
#print('finale shape ' ,labels.shape)

# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits
#data = data.reshape((data.shape[0], 224, 224))
#data = data[:, np.newaxis, :, :]
#(trainData, testData, trainLabels, testLabels) = train_test_split(
#	data / 255.0, labels, test_size=0.33)

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
#trainLabels = np_utils.to_categorical(trainLabels, 3)
#testLabels = np_utils.to_categorical(testLabels, 3)
labels = np_utils.to_categorical(labels, 3)
# initialize the optimizer and model

print("[INFO] compiling model...")
opt = SGD(lr=0.01)																																																																																																																																																																					
model = LeNet.build(width=224, height=224, depth=3, classes=3,
	weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
	print("[INFO] training...")
	model.fit(data/255.0,  labels, batch_size=32,epochs=10,
		verbose=1)

	# show the accuracy on the testing set
	#print("[INFO] evaluating...")
	#(loss, accuracy) = model.evaluate(testData, testLabels,
	#	batch_size=32, verbose=1)
	#print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

#randomly select a few testing digits
#==============================================================================
#for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
	# classify the digit
# 	probs = model.predict(testData[np.newaxis, i])
# 	prediction = probs.argmax(axis=1)
# 
# 	# resize the image from a 28 x 28 image to a 96 x 96 image so we
# 	# can better see it
# 	image = (testData[i][0] * 255).astype("uint8")
# 	image = cv2.merge([image] * 3)
# 	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
# 	cv2.putText(image, str(prediction[0]), (5, 20),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
# 
# 	# show the image and prediction
# 	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
# 		np.argmax(testLabels[i])))
# 	cv2.imshow("Digit", image)
# 	cv2.waitKey(0)
# 
#==============================================================================
