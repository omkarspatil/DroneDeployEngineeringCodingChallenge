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
<p>Also sorted all such contours(possible position markers) found by the area of the contour. We know that the QR code was printed on a white paper and we want to exclude the contour of the paper and any other polygons from being selected as a position marker. We can do this by selecting the smallest 3 contours by area as our final position markers.</p>

<p>Step 3.a : This is an optional step only for verification purposes. It draws the 3 identified position markers
             of the QR code(We add all children of the 3 contours too for completeness)</p>
<img src=""></img>

<p>Step 3.d :Determine the orientation of the QR code based on the distance of the outlier(top marker) from the line joining the other two markers and the slope of this line.(The orientation allows us to differentiate the bottom position marker from the right position marker.)</p>

<p> Step 4: Get the farthest corners of each marker from the center of the QRCode. Mark these corners as three corners of
            the QR code.
            corner A : TOP LEFT corner
            corner B : TOP RIGHT corner
            corner C : BOTTOM LEFT corner 
            corner D : BOTTOM RIGHT corner(derived from A,B and C)</p>
            
<p>Step 5: Use the 4 corners identified to find a 4-point perspective transform using OpenCV's solvePNP.</p>
<p>Step 5.a: Put the camera pixel co-ordinates of the corners w.r.t to the original image in a numpy array.</p>
<p>Step 5.b: Put the known 3D co-ordinates(world coordinates) of these points in a numpy array</p>
<p>Step 5.c: Use the iPhone 6's calibration matrix and distortion co-efficient matrix for OpenCV.
           Available here : https://stackoverflow.com/questions/14680944/create-opencv-camera-matrix-for-iphone-5-solvepnp</p>
<p>Step 5.d : Run the solvePNP method using flag cv2.SOLVEPNP_ITERATIVE</p>
<p>Step 6: Project a set of X,Y,Z axes on the QR code in the original image to check if the rotation and translation
            vectors are fairly accurate. Also draw them on the original image.</p>
<p>Step 7: Obtain a 3x3 rotation matrix from the 3 euler angles in the rotation vector returned from the solvePNP function
           Reference: https://www.chiefdelphi.com/forums/showthread.php?threadid=158739</p>
<p> Step 7.a : We form a 4x4 transformation matrix to transform from the camera coordinates to the world coordinates using the rotation and translation vector returned by solvePNP using the following relations:
<img src="https://www.cc.gatech.edu/~hays/compvision2016/results/proj3/html/agartia3/ProjectionMatrix.jpg"></img></p>

```python
ZYX, jac = cv2.Rodrigues(rotation_vector)
totalrotmax = np.array([[ZYX[0, 0], ZYX[0, 1], ZYX[0, 2], translation_vector[0]],
                        [ZYX[1, 0], ZYX[1, 1], ZYX[1, 2], translation_vector[1]],
                        [ZYX[2, 0], ZYX[2, 1], ZYX[2, 2], translation_vector[2]],
                        [0, 0, 0, 1]])
```

<p>Step 7.b : The previous step's matrix is the transformation matrix from world coordinates (centered on the target) to camera coordinates. (Centered on the camera) We need a matrix that transforms from camera coordinates to world. 
We hence compute the inverse of that matrix.
           
```python
inverserotmax = np.linalg.inv(totalrotmax)
```

</p>


<p>Step 7.c : We can now compute the yaw,pitch and roll values of the camera from the 3x3 submatrix of the above inverserotmax matrix</p>
<p>Step 7.d : Use the 3x3 rotation matrix from the computed inverse to draw the X,Y,Z axes of the camera in
             the visualization</p>
<p>Step 8: Generate the visualization in 3D using matplotlib</p>
<p>Step 8.a: Draw the QR code on the XY plane at (0,0,0) in the world</p>
<p>Step 8.b: Plot the position of the camera as a marker in the world co-ordinate system.</p>
<p>Step 8.c: Plot the rotated axes of the camera to get a sense of the rotation in 3D</p>
           



    
