###model.py file for CarND Project 3 - Behavior Cloning. 

import csv
import cv2
import numpy as np
import sklearn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

#Name of the model file.
model_file_name = "model.h5"

#Read the data from CSV file.
samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
		
#Total number of lines read
print('Total number of lines in the datafile is',len(samples))

#Spliting the total data into training and validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Total lines in training sample',len(train_samples))
print('Total lines in validation sample',len(validation_samples))

#Preprocessing the image - 
#Cropping out the non useful parts like Sky and Trees and only considered images of road.
#Resize the image to make the processing in the neural network faster.
#converting the image to rgb.
def preprocess(image):
    cropped = image[60:150,:,:]
    resized = cv2.resize(cropped, (32,32))
    rgb = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
    return rgb

#Generator will yield data in batches. Each batch will return not more than 192 images and labels. 
#6 types of images are considered to train the mode. Camera images - Center,Left and Right and flipped images of all 3. This is reduce the biasing of neural network on 1 side.
# and will also generate more data.
def generator(samples,batch_size = 32):
    num_samples = len(samples)
    correction = 0.2
	#location where data is stored on my local machine.
    addstr = 'C:/Users/shobhit.srivastava/CarND-P3/data/IMG/'
	
    while True:
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering = []
            
            for batch_sample in batch_samples:
                
                name_center = addstr + batch_sample[0].split('/')[-1].strip()
                image_center =  preprocess(cv2.imread(name_center))
                steering_center = float(batch_sample[3])
                
				#As per the training data, steering value 0 has outnumbered other steering values. Once not all will be considered to avoid making the neural network biased for 
				#steering value 0. The images with 0 steering value will be dropped by a probabilty of 40%.
                if(steering_center == 0):
                    drop = np.random.random_sample()
                    if(drop>0.6):
                        continue
                
				#Orginal images
                name_left = addstr + batch_sample[1].split('/')[-1].strip()
                image_left =  preprocess(cv2.imread(name_left))
                steering_left = steering_center + correction
                
                name_right = addstr + batch_sample[2].split('/')[-1].strip()
                image_right =  preprocess(cv2.imread(name_right))
                steering_right = steering_center - correction
                
                image_center_flipped =  np.fliplr(image_center)
                steering_center_flipped = -steering_center
                
				#Flipped images
                image_left_flipped =  np.fliplr(image_left)
                steering_left_flipped = -steering_left
                
                image_right_flipped =  np.fliplr(image_right)
                steering_right_flipped = -steering_right
                
				#Adding all 6 types of images in the list
                images.extend([image_center,image_left,image_right,image_center_flipped,image_left_flipped,image_right_flipped])
                steering.extend([steering_center,steering_left,steering_right,steering_center_flipped,steering_left_flipped,steering_right_flipped])
            
            X_train = np.array(images)
            y_train = np.array(steering)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#importing model and layers from keras
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,MaxPooling2D,Activation,Dropout
 
#The model is a sequential model. It has 2 Convolution2D layer and 3 Dense layer. 
model = Sequential()

#Normalizing the image to have near 0 mean. This will reduce the loss and help faster gradient descent.
model.add(Lambda(lambda x : x/255.0 - 0.5,input_shape=(32,32,3)))

#First convolution layer.
model.add(Convolution2D(32,5,5,border_mode='same',subsample=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

#Second convolution layer.
model.add(Convolution2D(64,5,5,border_mode='same',subsample=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

#Flattening the output to send to Dense layer
model.add(Flatten())

#First fully connected layer. Dropout to avoid overfitting.
model.add(Dense(128))
model.add(Dropout(0.2))

#Second fully connected layer. 
model.add(Dense(64))

#Final fully connected layer. Output is a single floating point number whic is value of steering angle.
model.add(Dense(1))

#Compiling model
#Loss function. As this is problem of linear regression, mean square error (mse) is used as a loss function.
#Optimizer - Adam optimizer is used as it is better than simple SGD. It takes care of decaying learning rate interntally.
model.compile(loss='mse', optimizer='adam')

#Calling generator to get yield values.
model.fit_generator(train_generator,samples_per_epoch = 6*len(train_samples),validation_data=validation_generator,nb_val_samples=len(validation_samples),nb_epoch=4)

#saving model file
model.save(model_file_name)
print('Model file -',model_file_name,' is saved sucessfully')