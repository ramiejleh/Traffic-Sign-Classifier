# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualization.png "Visualization"
[image2]: ./test_images/sign1.png "Traffic Sign 1"
[image3]: ./test_images/sign2.png "Traffic Sign 2"
[image4]: ./test_images/sign3.png "Traffic Sign 3"
[image5]: ./test_images/sign4.png "Traffic Sign 4"
[image6]: ./test_images/sign5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ramiejleh/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a random pick of the images in the dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to do a minmax normalizaion which is defined in the `preprocess()` function because training process is more accurate on smaller numbers so keeping the data normalized leads to a higher accuracy.

I decided to do some data augmentation to help the model generalize and prevent overfitting.

To augment the data i used the following tensorflow methods:

`tf.image.per_image_standardization()`
`tf.image.random_flip_up_down()`
`tf.image.random_flip_left_right()`
`tf.image.random_hue()`
`tf.image.random_brightness()`


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|
|:---------------------:|
| Input         		|
| Convolution        	|
| RELU					|
| Max pooling	      	|
| Local response normalization|
| Convolution   	    |
| RELU		            |
| Local response normalization|
| Max pooling	      	|
| Dropout	      	    |
| Fully connected		|
| RELU		            |
| Batch normalization   |
| Dropout	      	    |
| Fully connected		|
| RELU		            |
| Batch normalization   |
| Fully connected		|
|						|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Hyperparameters used:

 Parameter         		|     Value	    | 
|:---------------------:|:-------------:| 
| EPOCHS         		| 15   		    | 
| Batch size         	| 256 	        |           
| Learning rate			| 0.001	        |
| Optimizer	      	    | AdamOptimizer |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.953
* test set accuracy of 0.974
 
I started from the Lenet architecture which was taught in the CNN module on Udacity because it's a decent starting point but it will only give a validation accuracy of 0.87 which is not enough for the requirements here so i ahd to tweak it a bit.

The initial architecture was very simple and lacking some key features like batch normalization and dropout which help in preventing overfitting.

I added a couple of dropout layers and after consulting with the Tensorflow docs i added local response normalization layers after my convolutions. Plus i added batch normalization layers after two of my fully connected layers.

I also tuned the hyperparameters like number of increasing the number of epochs and batch size.

The model was decided to be a CNN because CNNs have proven their accuracy when dealing with images because a CNN looks at pixels in groups understanding more complicated patterns than a pixel-by-pixel approach.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The images should be pretty straight forward to classify as they were found on one of the official German traffic websites.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery road      	| Slippery road   								| 
| Speed limit (60km/h)  | Speed limit (60km/h) 				            |
| Pedestrians			| Pedestrians									|
| Traffic signals	    | Traffic signals				 				|
| Wild animals crossing	| Wild animals crossing    						|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.4%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For the first image, the model is clearly sure that this is a Slippery road sign (probability of 0.9), and the image does contain a Slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop sign   									| 
| .0002   				| Dangerous curve to the left 					|
| .00009				| No passing for vehicles over 3.5 metric tons	|
| .000002	      	    | Beware of ice/snow					 		|
| .000001				| Double curve      							|


For the second image, the model is clearly sure that this is a Speed limit (60km/h) sign (probability of 0.9), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Speed limit (60km/h)   						| 
| .0086   				| Speed limit (30km/h) 					        |
| .0081				    | Speed limit (50km/h)	                        |
| .00007	      	    | Speed limit (80km/h)					 		|
| .00001				| Wild animals crossing   						|

For the third image, the model is clearly sure that this is a Pedestrians sign (probability of 0.9), and the image does contain a Pedestrians sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .91         			| Pedestrians   						        | 
| .05   				| General caution					            |
| .02				    | Right-of-way at the next intersection	        |
| .00006	      	    | Traffic signals					 		    |
| .000009				| Road narrows on the right   					|

For the fourth image, the model is clearly sure that this is a Traffic signals sign (probability of 0.8), and the image does contain a Traffic signals sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .84         			| Traffic signals   						    | 
| .15   				| General caution					            |
| .0001				    | Pedestrians	                                |
| .0000004	      	    | Bicycles crossing					 		    |
| .0000003				| Road narrows on the right   					|

For the fifth image, the model is clearly sure that this is a Wild animals crossing sign (probability of 0.9), and the image does contain a Wild animals crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Wild animals crossing   						| 
| .0003   				| Road work					                    |
| .000001				| Dangerous curve to the left	                |
| .0000003	      	    | Double curve					 		        |
| .0000002				| Speed limit (50km/h)   					    |
