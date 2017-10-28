# DroneDeployEngineeringCodingChallenge
A computer vision based coding problem from DroneDeploy

<b>The problem statement</b>

This zip file https://www.dropbox.com/s/g67dmolq79ko4jk/Camera%20Localization.zip?dl=0 contains a number of images taken from different positions and orientations with an iPhone 6. Each image is the view of a pattern on a flat surface. The original pattern that was photographed is 8.8cm x 8.8cm and is included in the zip file. Write a Python program that will visualize (i.e. generate a graphic) where the camera was when each image was taken and how it was posed, relative to the pattern.

You can assume that the pattern is at 0,0,0 in some global coordinate system and are thus looking for the x, y, z and yaw, pitch, roll of the camera that took each image. Please submit a link to a Github repository contain the code for your solution. Readability and comments are taken into account too. You may use 3rd party libraries like OpenCV and Numpy.

<b> The approach used to solve the problem </b>
<p>Step 0   : Read the input image in grayscale using OpenCV</p>
<p>Step 1   : Runny OpenCV's Canny edge detection on the input image.</p>
<p>Step 2   : Run OpenCV's findContours to get all the contours and the contour hierarchy.</p>
<p>Step 3   : Find all the contours that are 4 sided polygons and if such a contour is found check if it encloses 5 or 
           more contours as its children. This technique works as we know that the position markers of a QR code is a square
           with 5 concentric squares(each border contributes to one square).</p>
<p>Step 3.a : Sorted all such contours(possible position markers) found by the area of the contour. We know that the QR code was printed on a white paper and we want to exclude the contour of the paper and any other polygons from being selected as a position marker. We can do this by selecting the smallest 3 contours by area as our final position markers.</p>
