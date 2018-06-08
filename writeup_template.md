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
[image7]: ./examples/image_augment.png "Flipped Image"

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network and fully connected layer, and the data is normalized in the model using a Keras lambda layer. 

![alt text][image3]

#### 2. Attempts to reduce overfitting in the model

The model contains Lambda layers and dropout in order to reduce overfitting, augment doesn't work well on my data set. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, Adam(lr=1e-4).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. To capture good driving behavior, I first recorded two laps on track one using center lane driving. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery from side. 

Then I repeated this process on track 1 by counter-wise in order to get more data points. 
Below is my data set distribution.

![alt text][image1]

To augment the data sat, I also flipped images and angles thinking that this would help to increase the data set and data variety For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 24465 number of data points. I think maybe aument helpfully, but it doesn't work as well as imagine, so doesn't used in this project.

![alt text][image2]

