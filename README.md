# Advanced Lane Finding

Project for the Self-Driving Car Engineer Nanodegree Program.

---

## Overview

[![](https://img.youtube.com/vi/4iMZfRZtLVo/0.jpg)](https://www.youtube.com/watch?v=4iMZfRZtLVo)

In this project I built an advanced lane-finding algorithm using distortion correction, image rectification, color transforms, and gradient thresholding. The algorithm is able to idenify lane curvature and vehicle displacement and overcame some environmental challenges such as shadows and pavement changes.

The goals/steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image01]: ./output_images/UndistChess.jpg "Undistorted"
[image02]: ./test_images/test2.jpg "Road Transformed"
[image03]: ./output_images/undst_test3.jpg "Road Transformed Undistorted"
[image04]: ./output_images/binary.png "Binary"
[image05]: ./output_images/src_image.png "Binary Masked"
[image06]: ./output_images/ROI.png "ROI"
[image07]: ./output_images/transf_color.png "transf_color"
[image08]: ./output_images/transf_binary.png "transf_binary"
[image09]: ./output_images/line.png "line"
[image10]: ./output_images/result.png "result"
[video1]: ./project_video_result.mp4 "Video"

## Dependencies

- Python 3.5
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [OpenCV](http://opencv.org/)
- [MoviePy](http://zulko.github.io/moviepy/)

## Code Description

### Camera Calibration

A brief state of how I computed the camera matrix and distortion coefficients.

---

The code for this step is contained in the file called `cam_cal.py`.

In this calibration, I used the snapshots of the classical black-white chessboard provided for calibration except for the `calibration1.jpg` image. This particular image was renamed as `test.jpg` for testing purpose.

The corners of each calibration image found with the help of `cv2.findChessboardCorners()` function are stored in the vector `imgpoints`. A vector named `objpoints` described the (x,y,z) coordinates where each chessboards corner should be. As we have a planar pattern, all Z coordinates were set to zero. Because, is a single pattern for all input images the same (x,y,z) coordinates was associate to every input image.

Then, a camera calibration was done using the `cv2.calibrateCamera()` function using `imgpoints` and the `objpoints` as the main parameters. The returned "camera matrix" and "distortion coeficients" were saved into a `calibration_parameters.p` file to be used in a future undistortion operation.

In order to visualize if the calibration was done correctly, using the `calibration_parameters.p` and the help of the `cv2.undistort()` function, the result obtained applying it to the `test.jpg` image was:

![alt text][image01]

### Pipeline

A description of how (and identify where in my code) I did this project.

---

#### 1. Distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image02]

The code for this step is contained in the file called `cam_cal_test.py`.

Broadly, reading each of the image in the `\test_images` folder apply the `cv2.undistort()` function with the `calibration_parameters.p` previusly calculated. Then ploted in a comparative way as this example shows:
![alt text][image03]

All of this images are saved in the `output_images\` folder named as `undst_test*.jpg`.

#### 2. Thresholded binary image

The code of this point can be found in the notebook `pipeline_img.ipynb` at the section named `STEP 3: Create a thresholding binary image`.

After several trials, I decide to use two thresholding colors and a gradient in an X direction. One thresholding color was intended to get the yellow lane line, the another one is to obtains the white lane line. The gradient was used to complement finding some fine details that can't be found by color. 

For the yellow line, I made a color space conversion from RGB to LUV and chose the V channel to apply a color threshold. I found that this channel is well suited for this proposed. 

For the white line, I made a color space conversion from RGB to HLS and chose the L channel to apply a color threshold.

Finally, for the gradient, an RGB to Grayscale was made to apply Sobel method and the absolute of the gradient in X was taken.

An example of the result is showing in the next figure: 
![alt text][image04]

In the color binary image, the red channel is the gradient, the green channel the attempt to get the yellow lane line and the blue one for white.


#### 3. Perspective transform 

The code for this section can be found in notebook `pipeline_img.ipynb` after the markdown cell with the title ' STEP 4: Bird-eye view transformation'.

As I used the `cv2.warpPerspective` function I need to define four source points and four destination points. For source points, I define each of the bottom corners of the image as points. The another two points are chosen to create a virtual parallel line with the lane lines and a little bit lower than the horizon. A visual inspection of this can be appreciated in this figure: 
![alt text][image05]

Because over the horizon and probably the data out of this border is useless. I apply a mask to ignore those pixels. I define a function to do this task called `region_of_interest(img, vertices)`. The code was taken from the first project of this nano degree. The result of applied this to the binary image is shown next:
![alt text][image06]

After extracting the region of interest and apply the transformation an example of the result is shown in the undistorted image and the binary image:
![alt text][image07]
![alt text][image08]

#### 4. Identified lane-line pixels and fit their positions with a polynomial

The code for this section can be found after the markdown cell with the title '# Step 5: Detect lane pixels' in the notebook `pipeline_img.ipynb`. 

The code of this section is the same used in the class of this topic in nano degree but is worth to note the hyperparameters that I tuned:

The number of sliding windows = 6. This was chosen considering for taking into account the dash lines. 

The width of the windows. Margin = 30. It's important to keep small to avoid the possible noise around the lines. 

The minimum number of pixels found to recenter window. 100 pixels was found to be a good choice as more pixels tend to ignore the curvature and less tend to add noise to straight lines.

#### 5. Calculat the radius of curvature of the lane and the position of the vehicle with respect to the center.

I did this in lines after the markdown cell title 'Step 6: Determine the curvature of the line and vehicle position respect to center' in my code in the `pipeline_img.ipynb`.

For the curvature of the line, I used the definition of the radius of curvature and the code showed in this course. I add some lines to estimate a virtual center curvature with interpolation as follow:
````
xp = [left_line, right_line]
fp = [left_curverad, right_curverad]
center_curverad = np.interp(lane_center, xp, fp)
````
, I think this is more convenient as input to control a car.

For units conversion, I measure how long in pixels the dashed line is (approx 73 pixels): 
![alt text][image09]

as well as the lane width (128 pixels). Knowing that the standard measures in meters are 3 m for the dashed line and 3.7 m for the lane width I defined the constant as follows:
```
ym_per_pix = 3/73 # meters per pixel in y dimension
xm_per_pix = 3.7/128 # meters per pixel in x dimension
```

For estimate the position of the car respect with to center I first got the bottom coordinates in the image for each line and considering that the center of the vehicle is in the center of the image:
```
left_line = leftx[0]
right_line = rightx[0]
vehicle_center = img.shape[1]/2
```

then I calculate a virtual lane center in x coordinates base on the lane width computed as the difference between each line position:
```
lane_width = right_line - left_line
lane_center = left_line + lane_width/2
```

finally, I get the offset with the difference between the vehicle center and the lane center. With this order, a positive offset mean that the vehicle is on the right of the center and a negative value that is on the left. To have a measurement in meters the difference is scaled by a factor of meters per pixels estimated in previous steps.  
```
offset = (vehicle_center - lane_center) * xm_per_pix
```

#### 6. Warp the detected line boundaries onto the original image.

I implemented this step in some lines after the markdown cell in the `pipeline_img.ipynb` titled 'Step 7: Warp the detected line boundaries onto the original image'

Here is an example of my result on a test image:

![alt text][image10]

#### 7. Pipeline applyed to video

The code for this step can be found in the `pipeline_img.ipynb`.


## Discussion

A brief discussion of problems / issues I faced in my implementation of this project.  Where will my pipeline likely fail?  What could I do to make it more robust?

---

I noticed that the perspective transformation distorts the lines (and objects) far from the vehicle. I find that is better shrink than stretch the points defined in the source points. Concretely, with a trapezoid source to be transformed to a rectangle is better to choose a rectangle with the same height of the trapezoid and width close to the length of the smaller base. If we choose a width close to or equal to the longest base, the lines -and more important the noise- are going to be wider. That gave me some problems on the early attempts to develop the project.

Another important issue is in the binarization where I found very helpful the exploration of different color spaces. 

Although the methodology proposed in this project is not perfect I think I know some things to try in order to do it better. The pipeline proposed was able to do well on the project_video but when I tested on the more challenge videos I learned some interesting things. The gradient in the thresholding binarization (at least in the way I calculated) add a lot of noise. Imperfections on the road for example. 
All parts matter, but I believe that the cornerstone of this, is in the sliding window step. For example in a single frame, sometimes lose the line because is unable to turn right or left. That lead to, in the worst case (most of the case in the harder_challenge_video.mp4), detecting noise as a line. The sliding window needs to have the ability to search in others directions and stop searching in some situations. But more than changing the direction, may be the window need to *rotate*. 

I hope that in the near feature have the opportunity the do it better, to do it more robust.

