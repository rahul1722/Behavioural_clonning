import csv
import numpy as np
import cv2
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def img_preprocess(image):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(gray, None)
    edges = cv2.erode(edges, None)
    new_image = edges[65:135,:,]
    new_image = cv2.resize(new_image, (320, 160))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB)
    
    return new_image

def generate_training_data(image_paths, angles, batch_size): #validation_flag=False):
    '''
    method for the model training data generator to load, process, and distort images, then yield them to the
    model. if 'validation_flag' is true the image is not distorted. also flips images with turning angle magnitudes of greater than 0.33, as to give more weight to them and mitigate bias toward low and zero turning angles
    '''
    image_paths, angles = shuffle(image_paths, angles)
    X = np.zeros((batch_size,160,320,3))
    y = np.zeros((batch_size,1))
    while True:       
        for i in range(batch_size):
            index = random.randint(1,19000)
            X[i] = img_preprocess(image_paths[index])
            y[i] = angles[index]
          
        yield X,y
        


"""
# Below lines are used to check the pre-processed image 

#plt.figure(figsize=(160,320))
#plt.imshow(im,cmap='gray')

#print(image.shape)
#print(image.size)

#modified_image = img_preprocess (image)

#from PIL import Image


image = cv2.imread('data/2.jpg',1)
#cv2.imshow('image',image)
new_im = img_preprocess(image)
filename = 'modified_image.jpg'
cv2.imwrite(filename,new_im)

#cv2.imshow('Modified image',modified_image)

"""

lines_udacity = []
lines_my_data = []
data = ["/opt/carnd_p3/data","my_data"] #Using both Udacity sample data and my data as a list

with open(data[0] + '/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    next(csv_file)
    for line in reader:
        lines_udacity.append(line)
       
images = []
measurements = []
correction = 0.5


for column in lines_udacity:
    for i in range(3):
        source_path = column[i]
        file_name = source_path.split('/')[-1]
        current_path = ("/opt/carnd_p3/data/IMG/"+ file_name)
        image = ndimage.imread(current_path)
        if i==0:
            measurement = float(column[3])
        elif i==1:
            measurement = float(column[3]) + correction
        elif i==2:
            measurement = float(column[3]) - correction 
        images.append(image)
        measurements.append(measurement)
    
        
with open(data[1] + '/my_data_driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    next(csv_file)
    for line in reader:
        lines_my_data.append(line)
        
for column in lines_my_data:
    for i in range(3):
        source_path = column[i]
        file_name = source_path.split('\\')[-1]
        current_path = ("my_data/my_data_IMG/"+ file_name)
        image = ndimage.imread(current_path)
        if i==0:
            measurement = float(column[3])
        elif i==1:
            measurement = float(column[3]) + correction
        elif i==2:
            measurement = float(column[3]) - correction 
        images.append(image)
        measurements.append(measurement)
 

images = np.array(images)
measurements = np.array(measurements)

print(images.shape)
print(measurements.shape)


Image_train, Image_test, angle_train, angle_test = train_test_split(images, measurements,test_size=0.2, random_state=42)



from keras.utils.data_utils import Sequence
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(24,(5,5), strides=(2,2), activation='relu'))
model.add(Convolution2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(Convolution2D(48,(5,5), strides=(2,2), activation='relu'))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3)) #added 2 dropout layers as the model was over-fitting without it
model.add(Dense(50))
model.add(Dropout(0.3)) 
model.add(Dense(10))
model.add(Dense(1))



train_data = generate_training_data(Image_train, angle_train, 1500) #1500 samples per generator data was selected on a random basis. I tried with 1000 but it wasn't learning quite well. when tried with 2000, the GPU would take a very long time. So I chose something in-between and it worked well
valid_data = generate_training_data(Image_train, angle_train, 750)
test_data = generate_training_data(Image_test, angle_test, 200)



model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_data, epochs=4, validation_data=valid_data, steps_per_epoch=200, validation_steps=200)

model.save('model.h5')

