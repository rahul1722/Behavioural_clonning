import cv2
import csv
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import random


def img_preprocess(image):
    #new_image = image.astype(float)
    #new_image = image[65:135,:,:]
    new_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    new_image = cv2.GaussianBlur(new_image,  (3, 3), 0)
    #new_image = cv2.resize(new_image, (200, 75))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_YUV2RGB)
    return new_image

def generate_training_data(image_paths, angles, batch_size): #validation_flag=False):
    '''
    method for the model training data generator to load, process, and distort images, then yield them to the
    model. if 'validation_flag' is true the image is not distorted. also flips images with turning angle magnitudes of greater than 0.33, as to give more weight to them and mitigate bias toward low and zero turning angles
    '''
    image_paths, angles = shuffle(image_paths, angles)
    X = np.zeros((batch_size,160,320,3))
    y = np.zeros((batch_size,1))
    #X = np.array()
    #y = np.array()
    while True:       
        for i in range(batch_size):
            index = random.choice(100)
            X[i] = img_preprocess(image_paths[index])
            y[i] = angles[index]
          
        yield X,y
        
lines = []


with open('/opt/carnd_p3/data/driving_log.csv') as csv_file:
    
    reader = csv.reader(csv_file)
    #csv.field_size_limit(15000)
    next(csv_file)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
correction = 0.2

for column in lines:
    for i in range(3):
        source_path = column[i]
        file_name = source_path.split('/')[-1]
        current_path = ('/opt/carnd_p3/data/IMG/'+file_name)
        image = ndimage.imread(current_path)
        if i==0:
            measurement = float(column[3])
        elif i==1:
            measurement = float(column[3]) - correction
        elif i==2:
            measurement = float(column[3]) + correction 
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
model.add(Cropping2D(cropping=((60,20), (0,0))))
model.add(Convolution2D(24,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(48,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Dropout(0.3))
#model.add(Convolution2D(76,3,3, activation='relu'))
#model.add(Convolution2D(88,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Dense(10))
#model.add(Dropout(0.5))
model.add(Dense(1))



train_data = generate_training_data(Image_train, angle_train, 70)#, validation_flag=False)
valid_data = generate_training_data(Image_train, angle_train, 20)#, validation_flag=True)
test_data = generate_training_data(Image_test, angle_test, 20)#, validation_flag=True)



model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_data, epochs=2, validation_data=valid_data, steps_per_epoch=30, validation_steps=20)

model.save('model1.h5')

 
    