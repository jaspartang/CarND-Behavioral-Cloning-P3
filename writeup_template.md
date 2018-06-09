# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_visualization.png "Data Visualization"
[image2]: ./examples/loss_epoch.png "Loss Epoch"
[image3]: ./examples/nvidia_network.png "Nvidia Network"
[image6]: ./examples/original_image.png "Normal Image"
[image7]: ./examples/image_flip.png "Flipped Image"
[image8]: ./examples/image_shift.png "shift Image"
[image9]: ./examples/image_augment.png "augment Image"

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network and fully connected layer, and the data is normalized in the model using a Keras lambda layer. 

![alt text][image3]

#### 2. Attempts to reduce overfitting in the model

The model contains Lambda layers and dropout in order to reduce overfitting, augment doesn't work well on my data set. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer adam.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. To capture good driving behavior, I first recorded two laps on track one using center lane driving. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery from side. 

Then I repeated this process on track one by counter-wise in order to get more data points. 
Below is my data set distribution.

![alt text][image1]

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia network, I thought this model might be appropriate as suggested.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training set but a low mean squared error on the validation set. This implied that the model was underfitting even with the dropout layers. 

To combat the underfitting, I modified the model, use the batch normalization, it's not a  hyperparameter, I like it.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I collecting more data from side to center, to taught the network learn from it.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.
_________________________________________________________________

Layer (type)                 Output Shape              Param #   
_________________________________________________________________
lambda_6 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_6 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 31, 158, 24)       1824      
_________________________________________________________________
batch_normalization_1 (Batch (None, 31, 158, 24)       96        
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 14, 77, 36)        21636     
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 77, 36)        144       
_________________________________________________________________
conv2d_28 (Conv2D)           (None, 5, 37, 48)         43248     
_________________________________________________________________
batch_normalization_3 (Batch (None, 5, 37, 48)         192       
_________________________________________________________________
conv2d_29 (Conv2D)           (None, 3, 35, 64)         27712     
_________________________________________________________________
batch_normalization_4 (Batch (None, 3, 35, 64)         256       
_________________________________________________________________
conv2d_30 (Conv2D)           (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_6 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_21 (Dense)             (None, 100)               211300    
_________________________________________________________________
activation_16 (Activation)   (None, 100)               0         
_________________________________________________________________
dense_22 (Dense)             (None, 50)                5050      
_________________________________________________________________
activation_17 (Activation)   (None, 50)                0         
_________________________________________________________________
dense_23 (Dense)             (None, 10)                510       
_________________________________________________________________
activation_18 (Activation)   (None, 10)                0         
_________________________________________________________________
dense_24 (Dense)             (None, 1)                 11        
_________________________________________________________________
Total params: 348,907
Trainable params: 348,563
Non-trainable params: 344
_________________________________________________________________


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.
I then recorded the vehicle recovering from the left side and right sides of the road back to center.
Then I repeated this process on track one by counter-wise in order to get more data points, but when testing on the simulator, the car partially left the track at some points, then capture more data recovery from side especially the fail points.

To augment the data set, I also augment images and angles, the car can stay at the center most of time, rest of time nearby the left side or right side; then I flip the images and angles, but I'm very be disappointed at the result.

At last, I shift the images, got amazing result. Below pictures show what it to be after fliped, shifted, and augmented.

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

After the collection process, I had 29610 number of data points. I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.I used an adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image2]

### Summary

In this project, I've trained a model to guide an autonomous car running in a simulator by End-to-End learning. The autonomous car owns three cameras: left, center and right, thus, all the images captured by these camera are used to train the model after some preprocess. The result show that it can perfected running in the simulator by itself.