# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

1.I converted the images to grayscale, then I applied Gaussian blur with parameters `kernel_size = 5`

![step1](./pipeline_images/1_Gaussian.png "Step 1: Apply Gaussian blur")

2.I applied Canny edge detection.  In line with the recommended 2:1 or 3:1 ratio of thresholds, I used `low_threshold = 50` and `high_threshold = 150`.

![step2](./pipeline_images/2_Canny.png "Step 2: Apply Canny edge detection")

3.I applied a mask to the image resulted from Canny image detection.  I used a simple polygon mask.

 ![step3](./pipeline_images/3_mask.png "Step 3: Apply a mask")


4.Applied the Hough Transform to the masked image from step 3.

![step4](./pipeline_images/4_Hough.png "Step 4: Hough Transform")

5.Process and filter the lines found by the Hough Transform.

![step5](./pipeline_images/5_weighted_img.png "Step 5: Filter, process, and select edges")

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by increasing the thickness to `10`.



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lanes are curved. This pipeline may lead poor results in urban roads with short turns. Second, It may struggle when there are traffic signs. This pipeline may not be robust to different lighting.  Finally, it would probably have difficulty when there are a lot of cars on the road and the lane lines cannot be seen 100+ feet in front of the car. 


### 3. Suggest possible improvements to your pipeline

To more accurately detect the lane lines, it would be beneficial to fit a nonlinear curve to the lanes (such as a spline), instead of fitting a line.  

One issue that can be seen in the videos is that the lane lines sometimes jump around.  To obtain smoother results, we could incorporate information from the previous frame(s) when detecting lanes in the current frame.
