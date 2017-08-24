## Writeup

---
** Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[vehicle_non-vehicle_image]: ./output_images/vehicle_non-vehicle.png
[vehicle_hog_image]: ./output_images/vehicle_hog_image.png
[non-vehicle_hog_image]: ./output_images/non-vehicle_hog_image.png
[original]: ./output_images/test1.jpg_original.png
[boxes]: ./output_images/test1.jpg_boxes.png
[heatmap]: ./output_images/test1.jpg_heatmap.png
[result]: ./output_images/test1.jpg_result.png
[boxes1]: ./output_images/boxes1.png
[boxes4]: ./output_images/boxes4.png
[boxes5]: ./output_images/boxes5.png
[heatmap1]: ./output_images/heatmap1.png
[heatmap4]: ./output_images/heatmap4.png
[heatmap5]: ./output_images/heatmap5.png
[result1]: ./output_images/result1.png
[result4]: ./output_images/result4.png
[result5]: ./output_images/result5.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][vehicle_non-vehicle_image]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block` ).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are examples for one random `vehicle` and `non-vehicle` using the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][vehicle_hog_image]
![alt text][non-vehicle_hog_image]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and choosed finally the following:

```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel='ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Linear Support Vector Classification using `LinearSVC` model of `sklearn` in the forth code cell of the IPython notebook. The evaluation of the estimator performance showed a test accuracy of SVC =  99.3%
I saved the trained model and parameters using `pickle` for the vehicle detection pipelne.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the function `find_cars` in the fifth code cell of the IPython notebook,  to extract features using hog sub-sampling and make predictions.  A sample output from the same is shown below.

![alt text][boxes]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][boxes1]
![alt text][boxes4]
![alt text][boxes5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  

To avoid false negative detection, meaning it has be ensured that a visible car is also detected, the window size is varied between scale=1.5, scale=2.0 and scale=2.5 and appropriately the start/stop values for the search area in y and the overlap ratio has been increased for the video.

To avoid false positives that are present only for 1-2 frames a method has been implemented, that uses multi-frame accumulated heatmap: the heatmap of the last 5 frames is stored and the  thresholding and labelling is done on the average of these heatmaps. To store the heatmaps I am using collections.deque. This techniqe provides stable bounding boxes as well. 

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are some frames and their corresponding heatmaps:

![alt text][heatmap1]
![alt text][heatmap4]
![alt text][heatmap5]


### Here the resulting bounding boxes are drawn onto some frames in the series:
![alt text][result1]
![alt text][result4]
![alt text][result5]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The developed pipeline may possibly fail in varied lighting and illumination conditions. Also, the multi-window search may be optimized further for better speed and accuracy.

