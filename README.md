# HOG Features for Human Detection using python from Scratch

## Introduction
Histogram of Oriented Gradients, also known as HOG, is a feature descriptor like the Canny Edge Detector, SIFT (Scale Invariant and Feature Transform) . It is used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in the localized portion of an image. This method is quite similar to Edge Orientation Histograms and Scale Invariant aFeature Transformation (SIFT). The HOG descriptor focuses on the structure or the shape of an object. It is better than any edge descriptor as it uses magnitude as well as angle of the gradient to compute the features. For the regions of the image it generates histograms using the magnitude and orientations of the gradient.
In this project we will use HOG Features for Human Detection, in this project we will use a relatively small dataset for simplicity purposes. We will use 20 train images (10 human images and 10 non-human images), and 10 test images (5 human images and 5 non-human images)

## Implementation Steps

### Step 1: Gradient computation
### Step 2: Orientation binning
### Step 3: Descriptor blocks
### Step 4: Block normalization
### Step 5: object Recognition using Similarity Score

## Results
The results of 10 test images with there HOG Discriptor, 3NN and classification is given below. We see that we are gettung two misclassificaitons (no_person_no_bike_258_cut
.bmp and no_person_no_bike_264_cut.bmp).

![image](https://user-images.githubusercontent.com/50113394/148706179-4aec0eba-b6e3-4d4a-9d98-a8975b9b2f5d.png)
