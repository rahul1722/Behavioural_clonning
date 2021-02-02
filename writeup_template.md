# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes for the first 4 layers and a 3x3 filter size for the 5th layer. The layers have depths between 24 and 64 (model.py lines 139-151) 

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 140). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 147 & 149). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 155-157). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 161).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to identify the different lane marking from the images and give a strearing angle as an output using a regression model.

My first step was to use a convolution neural network model similar to the NVIDIA model as mentioned it the classes. I thought this model might be appropriate because it is proven and I will only have to tweek the hyperparameters in order to build a good model. I was able to succefully complete 1 lap using this model without going off the road. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I intorduced 2 dropout layers and also played around with the batch size in generator function. Depending on the validation accuracy, I changed the batch size and finally 1500 samples per batch and 200 steps per epoch worked for me.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded additional data so the car can learn better with all the different patterns of lanes. This helped to bring back the car to the center whenever it is getting close to the lane markings.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 139-151) consisted of a convolution neural network with the following layers and layer sizes ...

Layer 1 (Convolution) - 78 x 158 x 24 ; Stride - 2 x 2 ; Filter - 5 x 5
Layer 2 (Convolution) - 37 x 77 x 36 ; Stride - 2 x 2 ; Filter - 5 x 5
Layer 3 (Convolution) - 17 x 37 x 48 ; Stride - 2 x 2 ; Filter - 5 x 5
Layer 4 (Convolution) - 6 x 17 x 64 ; Stride - 2 x 2 ; Filter - 3 x 3
Layer 5 Flatten layer
Layer 6 Dense - 100 neurons
Layer 7 Dropout - 0.3
Layer 8 Dense - 50 neurons
Layer 9 Dropout - 0.3
Layer 10 Dense - 10 neurons
Layer 11 Dense - 1



![alt text][https://github.com/rahul1722/Behavioural_clonning/blob/master/Images/Architecture.JPG]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the Udacity sample data to begin with.

I then recorded the vehicle recovering from the left side and right sides of the road back to center only at specific areas, where the car was drifting off of the road. I saved my data in the opt folder and Udacity data in the CarND folder. I called images from both data sets by defining their paths in a list. I added a correction component of +/- 0.5 to the left and right images to keep the vehicle in the center. In the end, I had close to 24000 images which was then split into train and validatiion sets with 80:20 ratio. 

I used a generator function to feed the model with random batch of 1500 images in order to save disk space. This way, the model trained on all the 19000 images from the training set with minimum disk sapce. 

Before feeding the images to the model, the generator function preprocessed them using the img_preprocess function. Images were pre processed using the dilute and errode methods from the cv2 class and cropped to show only the road and avoid all the background. I believe, this helped the model to train with very little data.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 with 200 steps per epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
