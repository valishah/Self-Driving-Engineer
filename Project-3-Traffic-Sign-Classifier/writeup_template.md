# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/image_sample.jpg "sample images"
[image2]: ./writeup_images/sign_types.jpg "sample counts"
[image3]: ./writeup_images/five_signs.jpg "5 signs"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/pabloppp/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

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

First of all I displayed some images, and figured out that the images are grouped in sets of 30 images belonging to the same sign, and that seem to come from the same video (that means that our 34799 sized dataset has only 1160 different traffic signs)

Here I show the first image of every set of 30 for 100 of those sets. 

![alt text][image1]

At first glance, with only 100 examples we already see some unbalance on the type of signs but let's plot the number of signs of every type:

![alt text][image2]

We can see that the number of samples for each sign is very unbalanced. This can produce a lot of overfitting because the network might just learn that some signs are less probable than other, and will think 'faling' on those is less bad than failing on others.

I finally decided to give it a try without data augmentation, just to see if we could get a 'good enough' model, that could be improved in the future with augmentation.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In order to preprocess the data I only performed 3 operations:

**First, I changed the images from RGB to YUV**  
I noticed that a lot of the images a a very low brighness, feeding the model with RGB values would force it to learn some kind of relation between the brightnes and the color, so I thought I could make the model learn faster by giving the brightness in a unique channel (the Y from YUV) that way, the colors (UV) are the same no matter what the brighness of the image are, and the model can learn both brightness and shape related features using the Y channel and color related features using the channels UV.

**Second, I equalized the values**  
We want to prevent a very bright or very dark pixel to affect our moddel, so I applied a CLAHE algorythm to 'flatten' the values, making the bright colors darker, and the dark colors brighter.

**Last, I reshaped the values from 0..255 to -1..1**  
This is a basic step to achieve 0 and equal variance.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet model as showed in the previous exercice, but I simply added some dropout (only actually used it in the 2 last fully connected layers)

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16 
| RELU					|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 										|
| Fully connected		| input flattened 5x5x16 (400) outputs 120,									|
| RELU				|  dropout 50%            									|
| Fully connected		| input 120 outputs 84 									|
| RELU				|  dropout 50%      
| Fully connected		| input 84 outputs 43

 
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used a learning rate of 0.003 (0.001 seemed to work, but learned a bit slow, 0.01 seemed to be to high and the model was missing the minimums) 

I am using the AdamOptimizer, so varying the learning rate a little bit shouldn't affect to much the final accuracy of the model.

BATCH_SIZE = 265  
EPOCHS = 20 (though 15 seem to be enough to get a pretty good accuracy)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.684%
* validation set accuracy of 96.961% 
* test set accuracy of 95.067%

My first approach was to run the LeNet model on the images without preprocessing. The validation accuracy was of around 89%, so I thought I might try tomake little improvements to achieve the extra 4% accuracy required to past this test, instead of completely changing the model.

First I performed the previously descrived image preprocessing, I achieved an accuracy of 93% (yay!), but the trainign accuracy was already as 99.9%.

Second I added added dropout to the last 2 fully connected layers to prevent overfitting, and achieved a validation accuracy of 96.9%.

I procedded to test it on the test dataset and achieved a 95% accuracy. 

This seems like overfitting on training/validation data, I think I could improve it by doing some data augmentation, and of course the model was barely touched so that might have another huge think that could improve.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3]

The 1st and 3rd image should be quite easy to classify, the 1st image is pretty obvious and for the the 3rd, even if it has a little bit of tilting, because of the nature of the sign it should be pretty invariant to rotation.

Image 2 is cut, so the classifier might have trubble finding it if it's looking for perfect round shapes + the white background can be confusing (also, after doing the tests I noticed that the arrow seems to be shaped differently to the one in the trainign dataset)

Image 4 is the hardest, it's even hard for a human to identify it because it's covered in snow.

Image 5 is tilted in the z azis, and because no data augmentation is done this probably could be an issue (the network might be looking for same sized numbers, for example)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry		| No entry   									| 
| Ahead only     			| Keep left 										|
| Roundabout mandatory				| Roundabout mandatory											|
| Slippery road	      		| Wild animals crossing					 				|
| 120km/h			| 70km/h      							|


As I expected, the model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 

I handpicked a specifically hard set of images, so I already expected the model to fail. I think this is cause by a lack of trainign examples (overfitting to the examples that we have). 
This accuracy could be easily improved by augmenting the dataset.

Except maybe for image 4, that one might never be guessed right XD

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

**No entry**  
- No entry 100.0%  
- Stop 0.0%  
- Speed limit (20km/h) 0.0%  
- Speed limit (30km/h) 0.0%  
- Priority road 0.0%  

**Ahead only**  
- Keep left 99.2%  
- Go straight or left 0.83%  
- Roundabout mandatory 0.01%  
- Turn right ahead 0.0%  
- No passing 0.0%  

**Roundabout mandatory**  
- Roundabout mandatory 99.991%  
- Keep left 0.01%  
- Go straight or left 0.0%  
- Keep right 0.0%  
- Turn left ahead 0.0%  

**Slippery road**  
- Wild animals crossing 16.5%  
- Bicycles crossing 12.3%  
- Speed limit (120km/h) 10.5%  
- Road narrows on the right 9.4%  
- Traffic signals 9.2%  

**Expected: Speed limit**  
- Speed limit (70km/h) 70.93%  
- Speed limit (20km/h) 20.5%  
- Speed limit (120km/h) 7.8%  
- Speed limit (50km/h) 0.4%  
- Speed limit (80km/h) 0.3%  

On image 1 and 3 the model doesn't seem to have any kind of doubt about the prediction.
Judging by this, it seems like the image 5 could be easily predicted by improving the dataset a little bit.

Image 4 as expect has a very high degree of uncertainty, the model is not sure of what the image is. That is in my opinion due to the bad conditions of the image (not even a human can be sure of what it is)

Predictions on image 2 are the most shocking, the model is pretty sure that it's a sign but is totally wrong, the correct prediction isn't even amonst the first 5. My take on this is (by looking again at the number of samples of each sign) that the expected sign has very few examples in the datase, the model has learnt that it's unlikely to be, and other similar signs (blue background, white arrow) have much more examples so the model is 'lazy' and instead of learning very accurate features it learns very broad ones, because the majority of the time he'll be right.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


