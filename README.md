# HOG Features for Human Detection using python from Scratch

<<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/149029808-2baeabda-937e-4b63-bb5e-3898d1328dcd.png" />
</p>

## Introduction
Histogram of Oriented Gradients, also known as HOG, is a feature descriptor like the Canny Edge Detector, SIFT (Scale Invariant and Feature Transform) . It is used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in the localized portion of an image. This method is quite similar to Edge Orientation Histograms and Scale Invariant aFeature Transformation (SIFT). The HOG descriptor focuses on the structure or the shape of an object. It is better than any edge descriptor as it uses magnitude as well as angle of the gradient to compute the features. For the regions of the image it generates histograms using the magnitude and orientations of the gradient.
In this project we will use HOG Features for Human Detection, in this project we will use a relatively small dataset for simplicity purposes. We will use 20 train images (10 human images and 10 non-human images), and 10 test images (5 human images and 5 non-human images)

## Implementation Steps

**Note**: Each image is first converted to grayscale using the formulae I = Round(0.299R + 0.587G + 0.114B) where R, G and B are the pixel values from the red, green and blue channels of the color image, respectively, and Round is the round off operator. 

### Step 1: Gradient computation

Take the input image you want to calculate HOG features of. Resize the image into an image of 160x96 pixels.
We have used the Prewitt’s operator for the computation of horizontal and vertical gradients. M[i, j] = root(Gx**2 + Gy**2) is used to compute gradient magnitude, where
Gx and Gy are the horizontal and vertical gradients. We normalize and round off the gradient magnitude to integers within the range [0, 255]. Next, gradient angle is computed usning tan**-1(Gy/Gx).

**Note**: For image locations where the templates go outside of the borders of the image, assign a value of 0 to both the gradient magnitude and gradient angle. Also, if both Gx and Gy are 0, assign a value of 0 to both gradient magnitude and gradient angle.

Normalized Gx, Gy and Magnitude Images are shown below. 

Gradient w.r.t x:  
<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/149027385-0a0763c5-1e07-4634-8cf9-0a41d5be3cfe.png" />
</p>

Gradient w.r.t x:  
<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/149027413-56a8f2c1-2d71-4ceb-b4a5-96f3a998f5f7.png" />
</p>

Gradient Magnitude:  
<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/149028920-d0ee62e8-c3af-4432-a9fd-a5b056f4fa3f.png" />
</p>

### Step 2: Orientation binning

After obtaining the gradient of each pixel, the gradient matrices (magnitude and angle matrix) are divided into 8x8 cells to form a block. For each block, a 9-point histogram is calculated. A 9-point histogram develops a histogram with 9 bins and each bin has an angle range of 20 degrees. Below we have a 9-bin histogram in which the values are allocated after calculations. Each of these 9-point histograms can be plotted as histograms with bins outputting the intensity of the gradient in that bin.

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/149028121-d4b89d74-64ba-41de-bd72-aed8c09a88aa.png" />
</p>

### Step 3: Descriptor blocks

With the given image size and the parameters given above for computing the HOG feature, there are 20 X 12 cells and 19 X 11 blocks in the detection window. The
dimension of the HOG feature vector is 7,524. (cell size = 8 x 8 pixels, block size = 16 x 16 pixels (or 2 x 2 cells), block overlap or step size = 8 pixels (or 1 cell.))

### Step 4: Block normalization

We then consider v to be the non-normalized vector containing all histogram in a given block. The following formula is used to normalize all the vectors to the number of bins considered. This normalization is done to reduce the effect of changes in contrast between images of the same object. From each block a 36-point feature vector is collected.

### Step 5: object Recognition using Similarity Score

Object Recogniton is done using 3NN classifier and similarity score. The distance between the input image and a training image is computed by taking the histogram intersection of their HOG feature vectors:

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/149029519-2829f2a3-4429-4190-9d45-15bf1ac5044e.png" />
</p>

## Results
The results of 10 test images with there HOG Discriptor, 3NN and classification is given below. We see that we are gettung two misclassificaitons (no_person_no_bike_258_cut
.bmp and no_person_no_bike_264_cut.bmp).

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/148706179-4aec0eba-b6e3-4d4a-9d98-a8975b9b2f5d.png" />
</p>
