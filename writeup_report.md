# **Behavioral Cloning** 

## Writeup Template


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/loss.png "Training and Validation Loss"
[image2]: ./examples/center_2017_05_07_21_17_10_578.jpg "Racing Lines"
[image3]: ./examples/center_2017_05_13_16_24_52_689.jpg "Center Line"
[image4]: ./examples/center_2017_05_13_19_08_38_439.jpg "Difficult Turn"
[image5]: ./examples/center_2017_05_13_19_08_53_616.jpg "Another Difficult Turn"
[image6]: ./examples/left_2017_05_13_19_11_16_464.jpg "Left Camera Image"
[image7]: ./examples/right_2017_05_13_19_09_15_326.jpg "Right Camera  Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is the same as one used by NVidia which is [described in detail here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) It consists of a convolution neural network with 3 layers with 5x5 filter sizes and depths between 24 and 48 followed by 2 layers with  (model.py lines 94-107) 

The model includes RELU layers to introduce nonlinearity (lines 95-99). There is also pre-processing layer which crops top 50 and bottom 20 pixels and does normalization using a Keras Lambda and Cropping2D layers (code lines 127-128).

In addition I have adapted a version of LeNet model but have abandoned it in favor of "NVNet". It can be re-introduced by changing line 130

#### 2. Attempts to reduce overfitting in the model

The model contains 3 dropout layers in order to reduce overfitting (model.py lines 102-106). 

The model was trained and validated on disjoint subsets of input data sets to ensure that the model was not overfitting (code line 115-117). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 133).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving,  driving with racing lines, additional training examples covering problematic sessions, driving the alternate course (center lane only) and all of this repeated on reverse course. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try existing image recognition architectures and modify them for steering angle regression instead of original classification task.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it employs convolution layers to extract basic and higher order features, which are being ran through couple of fully connected layers in order to capture dependencies between input features and target values. This sounds like neural network I expected to end up with, so it made sense to try it as starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. While low validation error didn't necessarily translate into good driving, it was a good indication of whether model is overfiting, so I could know if I need to address it. 

To combat the overfitting, I modified added 3 dropout layers between fully-connected layers and introduced single dropout rate parameter for quick tuning and experimentation. In addition, I augmented existing data by adding flipped images with inverted steering values, to eliminate steering bias. Also, I added images from side cameras. All that gave me 6 times the amount of images I had from actual training sessions. Depending on choice of steering correction factor for side camera images, it seemed like extra training samples are driving down the validation error rate.

Then I tuned dropout rate, batch size and number of epochs until it seemed that model is accurate and generalizes well.

The final step was to run the simulator to see how well the car was driving around track one. It decided it needs a pit stop after crossing the bridge. To improve the driving behavior, I added samples from training sessions that were focused on problem areas (such as taking sharp left turn after the bridge instead of driving into the pit). 

**HOWEVER:**

There was a discrepancy between image decoding in training script (using OpenCV) and driving script (using PIL) so model was trained using BGR encoded images while steering value evaluation was done on RGB encoded images. Surprisingly, this still worked fairly well sometimes, but models were unstable. By the time I discovered this problem, I ended up developing a very capable model architecture that was mostly hit-or-miss, and once I fixed this, driving suddenly got much better and more consistent. But I wasn't seeing gradual improvements in behavior every time I improved the architecture, so I can't tell what helped a lot and what didn't help much.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. It does so at maximum speed while resembling first-timer on racing track who just been taught racing lines in classroom. It does run over the stripes at few points, which is expected in performance driving as those often lay on fastest path around the track. It does manage to steer clear of the ledges which is what we want both for conservative and performance driving.

#### 2. Final Model Architecture

The final model architecture has been described above.

#### 3. Creation of the Training Set & Training Process

To capture fast driving behavior, I first recorded two laps on track one using approximate racing line driving. Here is an example image of racing line driving:

![alt text][image2]

Then, I recorded 2 laps of center line driving. Here is an example image of driving in the middle of the road.

![alt text][image3]

Further, after running the simulation and identifying problem areas, I recorded the vehicle doing the right thing at the spots where it was running off the road, so that the vehicle would learn to negotiate those turns successfully. These images show what a difficult turns look like:

![alt text][image4]
![alt text][image5]

Then I added center line session on second track in order to make model generalize outside of the first track.

To augment the data sat, I also flipped images and angles thinking this should eliminate steering bias. Further, I just added all of the side camera images with corrected steering angle, expecting this to be a substitute for recovery trajectory samples. This has created large training set (6 times the size of non-augmented set) and model that keeps the car on the road but has tendency to over-correct deviations from training trajectories.

It turned out that better use for side camera images was as substitute for certain portion of overrepresented samples with low steering angles. I substituted 90% of said images with either left or right camera version of same sample and appropriately adjusted steering angle. Result was reduced bias against sharp steering angles without much weaving from over-correcting. Here are some side-cam images.

![alt text][image6]
![alt text][image7]

After the collection process, I had 27360 data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by this graph.

![alt text][image1]

I used an adam optimizer so that manually training the learning rate wasn't necessary.


