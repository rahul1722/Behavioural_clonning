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
"""
def download(url, file):
    if not os.path.isfile(file):
        print("Download file... " + file + " ...")
        urlretrieve(url,file)
        print("File downloaded")

download('https://drive.google.com/drive/folders/1KkwnvcR1rZpd0X0BQussLkHE6IRrXwDA?usp=sharing','data') #s3 path of the dataset provided by udacity

print("All the files are downloaded")


def uncompress_features_labels(dir,name):
    if(os.path.isdir(name)):
        print('Data extracted')
    else:
        with ZipFile(dir) as zipf:
            zipf.extractall('data')
uncompress_features_labels('data.zip','data')


def data_Files(mypath):
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    print(onlyfiles)

print('All files downloaded and extracted')
"""

def random_distort(img, angle):
    
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255 
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor
    # randomly shift horizon
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), angle)


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
    #X = np.zeros((batch_size,160,320,3))
    #y = np.zeros((batch_size,1))
    #X = np.array()
    #y = np.array()
    while True:       
        for i in range(batch_size):
            #index = random.randint(1,len(image_paths))
            img = img_preprocess(image_paths[i])
            angle = angles[i]
            
            if abs(angle) > 0.2:
                #X.append(img)
                #y.append(angle)
                img = cv2.flip(img,1)
                angle = angle * -1
            #X.append(img)
            #y.append(angle)
            yield img,angle
        
            #img = img_preprocess(img)
            #angle = angles[index]
    
            #if not validation_flag:
                #img, angle = random_distort(img, angle)
            
            #X.append(img)
            #y.append(angle)
            #if len(X) == batch_size:
                #break
       
        X, y = ([],[])
        #image_paths, angles = shuffle(image_paths, angles)
        # flip horizontally and invert steer angle, if magnitude is > 0.33
"""   
        if abs(angle) > 0.33:
            img = cv2.flip(img, 1)
            angle *= -1
            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                break
        yield (np.array(X), np.array(y))
                #X, y = ([],[])
                #image_paths, angles = shuffle(image_paths, angles)
"""
    
def generate_training_data_for_visualization(images, angles, batch_size=20, validation_flag=False):
    '''
    method for loading, processing, and distorting images
    if 'validation_flag' is true the image is not distorted
    '''
    X = []
    y = []
    image_paths, angles = shuffle(images, angles)
    for i in range(batch_size):
        #print (image_paths[i])
        img = image_paths[i]
        angle = angles[i]
        img = img_preprocess(img)
        if not validation_flag:
            img, angle = random_distort(img, angle)
        X.append(img)
        y.append(angle)
    return (np.array(X), np.array(y))

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
        #file_name_left = source_path_left.split('\\')[-1]
        #current_path_left = ('data/IMG/' + file_name_left)
        #image_left = ndimage.imread(current_path_left)
        #images.append(image_left)
        #file_name_right = source_path_right.split('\\')[-1]
        #current_path_right = ('data/IMG/' + file_name_right)
        #image_right = ndimage.imread(current_path_right)
        #images.append(image_right)
"""
    correction = 0.0
    measurement_center = float(column[3])
    measurement_left = measurement_center + correction
    measurement_right = measurement_center - correction
    measurements.append(measurement_center)
    measurements.append(measurement_left)
    measurements.append(measurement_right)

   
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    #augmented_images.append(cv2.flip(image,1))
    #augmented_measurements.append(measurement*-1.0)
    image = img_preprocess(image)
    augmented_images.append(image)
    augmented_measurements.append(measurement)
"""    

images = np.array(images)
measurements = np.array(measurements)

print(images.shape)
print(measurements.shape)


Image_train, Image_test, angle_train, angle_test = train_test_split(images, measurements,test_size=0.05, random_state=42)

#test_image,test_angle = generate_training_data_for_visualization(Image_train, angle_train)

#print(Image_train.dtype)
#print(angle_train.dtype)

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

#bs = 30

train_data = generate_training_data(Image_train, angle_train, batch_size = 70)#, validation_flag=False)
valid_data = generate_training_data(Image_train, angle_train, batch_size = 20)#, validation_flag=True)
test_data = generate_training_data(Image_test, angle_test, batch_size = 20)#, validation_flag=True)

#print(train_data.type)
#print(valid_data.shape)
#print(test_data.shape)

model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_data, epochs=2, validation_data=valid_data, steps_per_epoch=30, validation_steps=20)

model.save('model.h5')

 
    