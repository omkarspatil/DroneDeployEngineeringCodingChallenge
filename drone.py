import cv2
import numpy as np
import operator
import math
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
import mpl_toolkits.mplot3d.art3d as art3d

'''Author : Omkar Patil 
   Email  : patilo@purdue.edu '''
'''Define possible QR code orientations that one can encounter in an image'''
CV_QR_NORTH = 0
CV_QR_EAST = 1
CV_QR_SOUTH = 2
CV_QR_WEST = 3

'''Given real world dimension of the QR code in cm.'''
QR_CODE_WORLD_SIZE=8.8

def main(input_image_paths):
    '''Allowed numpy to print complete arrays in the console to ease debugging'''
    np.set_printoptions(threshold=np.nan)
    for imgpath in images:
            '''Read the input image in grayscale'''
            img = cv2.imread(imgpath,0)

            '''Read the original image to draw corners of QRcode after they're identified and project the 3D axes.'''
            imgOuput = cv2.imread(imgpath)

            '''Step 1: Runny OpenCV's Canny edge detection on the input image.'''
            edges = cv2.Canny(img, 100, 200)

            '''Step 2: Run OpenCV's findContours to get all the contours and the contour hierarchy.'''
            image2, cnts2, heir2 = cv2.findContours(edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            '''Step 3: Find all the contours that are 4 sided polygons and if such a contour is found check if it encloses 5 or 
            more contours as its children. This technique works as we know that the position markers of a QR code is a square
            with 5 concentric squares(each border contributes to one square)'''
            markers={}
            for i in range(0,len(cnts2)):
                approx = cv2.approxPolyDP(cnts2[i], 0.02*cv2.arcLength(cnts2[i],True), True)
                if len(approx)==4:
                    k=i
                    c=0
                    while heir2[0][k][2] != -1:
                        k = heir2[0][k][2]
                        c = c + 1
                    if heir2[0][k][2] != -1:
                        c = c + 1
                    if c ==5:
                        markers[i] = cv2.contourArea(cnts2[i])

            #print len(markers)
            '''Sorted all such contours(possible position markers) found by the area of the contour. We know that the QR code was 
            printed on a white paper and we want to exclude the contour of the paper and any other polygons from being selected 
            as a position marker. We can do this by selecting the smallest 3 contours by area as our final position markers.'''
            markers = sorted(markers.items(), key=operator.itemgetter(1))
            #print markers
            indices_list=[]
            count=3
            for pair in markers:
                indices_list.append(pair[0])
                #print pair[0]
                count=count-1
                if count==0:
                    break

            '''Step 3.a : This is an optional step only for verification purposes. It draws the 3 identified position markers
             of the QR code(We add all children of the 3 contours too for completeness)'''
            new_list=[]
            for index in indices_list:
                current_contour = index
                while heir2[0][current_contour][2]!=-1:
                    new_list.append(cnts2[heir2[0][current_contour][2]])
                    current_contour = heir2[0][current_contour][2]

            mask = np.zeros(img.shape, dtype="uint8")
            cv2.drawContours(mask, new_list, -1, (255), 2)

            mask2 = np.zeros(img.shape, dtype="uint8")

            '''Step 3.b : Use OpenCV contour moments to find out the centers of the three position markers'''
            mu={}
            mc={}
            for index in indices_list:
                mu[index] = cv2.moments(cnts2[index])
                mc[index] = (int(mu[index]["m10"] / mu[index]["m00"]),int(mu[index]["m01"] / mu[index]["m00"]))


            '''Step 3.c : We identify the orientation of the QR code as pictured in the image(North,South,East,West).
            This is of importance to us when we visualize the pose of the phone camera with respect to the QR code'''
            A = indices_list[0]
            B = indices_list[1]
            C = indices_list[2]

            AB = cvDistance(mc[A], mc[B])
            BC = cvDistance(mc[B], mc[C])
            CA = cvDistance(mc[C], mc[A])

            outlier = 0
            median1 = 0
            median2 = 0

            '''Identify the top position marker as it can always be differentiated from the other two markers'''
            if AB > BC and AB > CA:
                outlier = C
                median1 = A
                median2 = B

            elif CA > AB and CA > BC:
                outlier = B
                median1 = A
                median2 = C

            elif BC > AB and BC > CA:
                outlier = A
                median1 = B
                median2 = C

            top = outlier

            '''Step 3.d :Determine the orientation of the QR code based on the distance of the outlier(top marker) from the line joining the
             other two markers and the slope of this line.'''
            dist = cvLineEquation(mc[median1], mc[median2], mc[outlier])
            (slope,align) = cvLineSlope(mc[median1], mc[median2])

            #print slope,align

            '''The orientation allows us to differentiate the bottom position marker from the right position marker.'''
            bottom=0
            right=0
            orientation=0

            if align == 0:
                right = median1
                bottom = median2
            elif slope < 0 and dist < 0:
                right = median1
                bottom = median2
                orientation = CV_QR_NORTH
            elif slope > 0 and dist < 0:
                bottom = median1
                right = median2
                orientation = CV_QR_EAST
            elif slope < 0 and dist > 0:
                bottom = median1
                right = median2
                orientation = CV_QR_SOUTH
            elif slope > 0 and dist > 0:
                right = median1
                bottom = median2
                orientation = CV_QR_WEST

            print "Orientation of QR code is " +str(orientation)

            '''Step 4: Get the farthest corners of each marker from the center of the QRCode. Mark these corners as three corners of
            the QR code.
            corner A : TOP LEFT corner
            corner B : TOP RIGHT corner
            corner C : BOTTOM LEFT corner 
            corner D : BOTTOM RIGHT corner(derived from A,B and C)'''

            center= mid_point(mc[median1],mc[median2])
            #print center
            #top
            cornerA = farthestFromP(getMarkerPoints(top, cnts2),center)

            #bottom
            cornerB = farthestFromP(getMarkerPoints(right, cnts2),center)

            #right
            cornerC = farthestFromP(getMarkerPoints(bottom, cnts2),center)

            '''Corner D can be found by treating it as the fourth point of a parallelogram resulting from A,B,C,D'''
            cornerD = getParallelogram4th(cornerB,cornerC,cornerA)
            #print "Corner D is " + str(cornerD)


            '''Mark the 4 corners identified on the original image for verification'''
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(imgOuput, cornerA, 8, (255, 0, 0), -1)
            cv2.putText(imgOuput, "A", cornerA, font, 3, (51, 153, 255), 10, cv2.LINE_AA)
            cv2.circle(imgOuput, cornerB, 8, (255, 0, 0), -1)
            cv2.putText(imgOuput, "B", cornerB, font, 3, (51, 153, 255), 10, cv2.LINE_AA)
            cv2.circle(imgOuput, cornerC, 8, (255, 0, 0), -1)
            cv2.putText(imgOuput, "C", cornerC, font, 3, (51, 153, 255), 10, cv2.LINE_AA)
            cv2.circle(imgOuput, cornerD, 8, (255, 0, 0), -1)
            cv2.putText(imgOuput, "D", cornerD, font, 3, (51, 153, 255), 10, cv2.LINE_AA)


            cv2.drawContours(mask2, new_list, -1, (255), 2)


            '''Step 5: Use the 4 corners identified to frind a 4-point perspective transform using OpenCV's solvePNP.'''

            '''Step 5.a: Put the camera pixel co-ordinates of the corners w.r.t to the original image in a numpy array.'''
            image_points = np.array([
                cornerA,  # Nose tip
                cornerB,  # Chin
                cornerC,  # Left eye left corner
                cornerD,  # Right eye right corne
            ], dtype="double")

            '''Step 5.b: Put the known 3D co-ordinates(world coordinates) of these points in a numpy array'''
            model_points = np.array([
                (0.0, QR_CODE_WORLD_SIZE, 0.0),  # Nose tip
                (QR_CODE_WORLD_SIZE, QR_CODE_WORLD_SIZE, 0.0),  # Chin
                (0.0, 0.0, 0.0),  # Left eye left corner
                (QR_CODE_WORLD_SIZE, 0.0, 0.0),  # Right eye right corne
            ])

            '''Step 5.c: Use the iPhone 6's calibration matrix and distortion co-efficient matrix for OpenCV.
             Available here : https://stackoverflow.com/questions/14680944/create-opencv-camera-matrix-for-iphone-5-solvepnp'''
            cx = img.shape[0]/float(2)
            cy = img.shape[1]/float(2)
            fx = 3288.47697
            fy = 3078.59787
            dist = np.array([-7.416752e-02, 1.562157e+00, 1.236471e-03, 1.237955e-03, - 5.378571e+00], dtype="double")
            camera_matrix = np.array(
                [[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]], dtype="double"
            )

            '''Step 5.d : Run the solvePNP method using flag cv2.SOLVEPNP_ITERATIVE'''

            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist, flags=cv2.SOLVEPNP_ITERATIVE)
            print "Rotation Vector:\n {0}".format(rotation_vector)
            print "Translation Vector:\n {0}".format(translation_vector)


            '''Step 6: Project a set of X,Y,Z axes on the QR code in the original image to check if the rotation and translation
            vectors are fairly accurate. Also draw them on the original image.'''
            (zAxis, jacobian)=cv2.projectPoints(np.array([(0.0, 0.0, QR_CODE_WORLD_SIZE*2)]), rotation_vector, translation_vector, camera_matrix,
                              dist)
            (yAxis, jacobian) = cv2.projectPoints(np.array([(0.0, QR_CODE_WORLD_SIZE*2, 0.0)]), rotation_vector, translation_vector,
                                                  camera_matrix,
                                                  dist)
            (xAxis, jacobian) = cv2.projectPoints(np.array([(QR_CODE_WORLD_SIZE*2, 0.0, 0.0)]), rotation_vector, translation_vector,
                                                  camera_matrix,
                                                  dist)

            cv2.line(imgOuput, cornerC, (long(yAxis[0][0][0]), long(yAxis[0][0][1])), (255, 0, 0), 8)
            cv2.line(imgOuput, cornerC, (long(zAxis[0][0][0]), long(zAxis[0][0][1])), (0, 255, 0), 8)
            cv2.line(imgOuput, cornerC, (long(xAxis[0][0][0]), long(xAxis[0][0][1])), (0, 0, 255), 8)


            '''Step 7: Obtain a 3x3 rotation matrix from the 3 euler angles in the rotation vector returned from the solvePNP function
            Reference: https://www.chiefdelphi.com/forums/showthread.php?threadid=158739'''

            ZYX, jac = cv2.Rodrigues(rotation_vector)
            #print ZYX

            '''Step 7.a : We form a 4x4 transformation matrix using homogenous coordinates'''
            totalrotmax = np.array([[ZYX[0, 0], ZYX[0, 1], ZYX[0, 2], translation_vector[0]],
                                    [ZYX[1, 0], ZYX[1, 1], ZYX[1, 2], translation_vector[1]],
                                    [ZYX[2, 0], ZYX[2, 1], ZYX[2, 2], translation_vector[2]],
                                    [0, 0, 0, 1]])

            '''Step 7.b : The previous step's matrix is the transformation matrix from world coordinates (centered on the target) to camera 
            coordinates. (Centered on the camera) We need a matrix that transforms from camera coordinates to world. 
            We hence compute the inverse of that matrix.'''
            inverserotmax = np.linalg.inv(totalrotmax)

            #print(inverserotmax)

            '''Step 7.c : We can now compute the yaw,pitch and roll values of the camera from the 3x3 submatrix of the above inverserotmax matrix'''
            yaw = math.degrees(math.atan2(inverserotmax[1, 0], inverserotmax[0, 0]))
            pitch = math.degrees(math.atan2(-1*inverserotmax[2, 0], math.sqrt(math.pow(inverserotmax[2, 1],2) + math.pow(inverserotmax[2, 2],2))))
            roll = math.degrees(math.atan2(inverserotmax[2, 1], inverserotmax[2, 2]))

            print 'Yaw:' + str(yaw)
            print 'Pitch:' + str(pitch)
            print 'Roll:' + str(roll)

            cameraPosition = -np.matrix(ZYX).T * np.matrix(translation_vector)
            print 'Camera is located at(world co-ordinates) : '+str(cameraPosition)
            camfordist = np.array(cameraPosition).ravel()
            abs_distance = cvDistance3D((QR_CODE_WORLD_SIZE/float(2),QR_CODE_WORLD_SIZE/float(2),0),(camfordist[0],camfordist[1],camfordist[2]))
            print 'Camera\'s distance from pattern:' + str(abs_distance) + "cm"

            '''Step 7.d : Use the 3x3 rotation matrix from the computed inverse to draw the X,Y,Z axes of the camera in
             the visualization'''
            inverserot = inverserotmax[0:3, 0:3]
            #print inverserot

            '''This computes just rotates the end points of the axes of the camera, we then translate it to the by the camera's
            co-ordinates to position it on the camera'''
            axesObjZ = np.dot(inverserot,np.matrix([0,0,10]).T)
            axesObjX = np.dot(inverserot,np.matrix([10,0,0]).T)
            axesObjY = np.dot(inverserot,np.matrix([0,10,0]).T)




            '''Step 8: Generate the visualization in 3D using matplotlib'''
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')

            '''Step 8.a: Draw the QR code on the XY plane at (0,0,0) in the world'''
            p = Rectangle((0, 0), QR_CODE_WORLD_SIZE, QR_CODE_WORLD_SIZE, angle=0.0,fill=True)
            ax1.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
            ax1.text(0, 0, 0, "C");
            ax1.text(QR_CODE_WORLD_SIZE, 0, 0, "D");
            ax1.text(QR_CODE_WORLD_SIZE, QR_CODE_WORLD_SIZE, 0, "B");
            ax1.text(0, QR_CODE_WORLD_SIZE, 0, "A");


            '''Step 8.b: Plot the position of the camera as a marker in the world co-ordinate system.'''
            cameraPosition = np.asarray(cameraPosition).ravel()
            ax1.plot([cameraPosition[0]], [cameraPosition[1]], [cameraPosition[2]], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=1)

            '''Step 8.c: Plot the rotated axes of the camera to get a sense of the rotation in 3D'''
            VecStart_x_Zaxis = [cameraPosition[0]]
            VecStart_y_Zaxis = [cameraPosition[1]]
            VecStart_z_Zaxis = [cameraPosition[2]]


            VecEnd_x__Zaxis = [axesObjZ[0] + cameraPosition[0]]
            VecEnd_y__Zaxis = [axesObjZ[1] + cameraPosition[1]]
            VecEnd_z__Zaxis = [axesObjZ[2] + cameraPosition[2]]

            VecEnd_x__Yaxis = [axesObjY[0] + cameraPosition[0]]
            VecEnd_y__Yaxis = [axesObjY[1] + cameraPosition[1]]
            VecEnd_z__Yaxis = [axesObjY[2] + cameraPosition[2]]

            VecEnd_x__Xaxis = [axesObjX[0] + cameraPosition[0]]
            VecEnd_y__Xaxis = [axesObjX[1] + cameraPosition[1]]
            VecEnd_z__Xaxis = [axesObjX[2] + cameraPosition[2]]


            ax1.plot(VecStart_x_Zaxis + VecEnd_x__Zaxis, VecStart_y_Zaxis + VecEnd_y__Zaxis, VecStart_z_Zaxis + VecEnd_z__Zaxis,color="green")

            ax1.plot(VecStart_x_Zaxis + VecEnd_x__Yaxis, VecStart_y_Zaxis + VecEnd_y__Yaxis,
                     VecStart_z_Zaxis + VecEnd_z__Yaxis,color="blue")

            ax1.plot(VecStart_x_Zaxis + VecEnd_x__Xaxis, VecStart_y_Zaxis + VecEnd_y__Xaxis,
                     VecStart_z_Zaxis + VecEnd_z__Xaxis,color="red")

            center_x = [QR_CODE_WORLD_SIZE/float(2)]
            center_y = [QR_CODE_WORLD_SIZE/float(2)]
            center_z = [0]

            xZ = [cameraPosition[0]]
            yZ = [cameraPosition[1]]
            zZ = [cameraPosition[2]]

            ax1.plot(center_x + xZ, center_y + yZ,center_z + zZ, color="black", linewidth=0.1, label="Distance from pattern = "
            +str(abs_distance)+"cm \n Yaw: "+ str(round(yaw,2)) + "deg" + " Pitch: "+str(round(pitch,2)) +"deg" + " Roll : "+str(round(roll,2))+"deg")

            '''Use a function to scale all i.e X,Y,Z equally in matplotlib
            Refer to : https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to'''
            set_axes_equal(ax1)
            plt.legend(loc='best')
            #plt.show()

            plt.savefig('visualizations/visual'+imgpath.split(".")[0].split("test_images/")[1]+".png")
            cv2.imwrite("markers/output" + imgpath.split("test_images/")[1], mask)
            cv2.imwrite("corners_and_axes/output_corners" +imgpath.split("test_images/")[1], imgOuput)

def getParallelogram4th(cornerB,cornerC,cornerA):
     '''Gets the fourth point of a parallelogram given 3 points'''
     mid= mid_point(cornerB, cornerC)
     return (int(2*mid[0] - cornerA[0]),int(2*mid[1]-cornerA[1]))


def getLargestContour(cnts):
    '''Gets the largest contour by area out of a given list of contours.'''
    best = 0
    maxsize = 0
    count = 0
    for cnt in cnts:
        if cv2.contourArea(cnt) > maxsize:
            maxsize = cv2.contourArea(cnt)
            best = count
        count = count + 1
    return best

def mid_point(point_X, point_Y):
    '''Computes the midpoint of the line segment joining two points'''
    return [(point_X[0] + point_Y[0]) / 2, (point_X[1] + point_Y[1]) / 2]

def cvDistance(p, q):
    '''Computes the euclidean distance between two points.'''
    return pow((pow(abs(p[0] - q[0]),2) + pow(abs(p[1] - q[1]),2)),0.5)

def cvDistance3D(p, q):
    '''Computes the euclidean distance between two points.'''
    return pow((pow(abs(p[0] - q[0]),2) + pow(abs(p[1] - q[1]),2) + pow(abs(p[2] - q[2]),2)),0.5)

def cvLineEquation(L,M,J):
    '''Finds the distance of a point J from a line joining two points L and M'''
    a = -1*((float)(M[1] - L[1]) / (M[0] - L[0]))
    b = 1.0
    c = (((float)(M[1]- L[1]) /(M[0] - L[0])) * L[0]) - L[1]
    #Now that we have a, b, c from the equation ax + by + c, time to substitute (x,y) by values from the Point J
    pdist = (a * J[0] + (b * J[1]) + c) / pow((a * a) + (b * b),0.5)
    return pdist

def cvLineSlope(L, M):
    '''Returns the slope of a line joining two points.'''
    dx=float(M[0]) - L[0]
    dy=float(M[1]) - L[1]
    if dy!=0:
        return ((dy/dx),1)
    else:
        return (0.0,0)

def getMarkerPoints(index,cnts2):
    '''Returns all the corners of a position marker contour identified.'''
    rect = cv2.minAreaRect(cnts2[index])
    box = cv2.boxPoints(rect)
    '''cnt = cnts2[index]

    cv2.circle(mask, (box[0][0], box[0][1]), thickness, (255, 255, 255), -1)
    cv2.circle(mask, (box[1][0], box[1][1]), thickness+2, (255, 255, 255), -1)
    cv2.circle(mask, (box[2][0], box[2][1]), thickness+4, (255, 255, 255), -1)
    cv2.circle(mask, (box[3][0], box[3][1]), thickness+6, (255, 255, 255), -1)'''

    return [(box[0][0], box[0][1]),(box[1][0], box[1][1]),(box[2][0], box[2][1]),(box[3][0], box[3][1])]


def farthestFromP(listofPoints,targetPoint):
    '''Returns the farthest point out of a list of points from a given target point.'''
    maxDistance=0
    maxPoint=None
    for point in listofPoints:
        if cvDistance(point,targetPoint) > maxDistance:
            maxDistance =cvDistance(point,targetPoint)
            maxPoint = point
    return maxPoint

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if __name__ == "__main__":
    images = ['test_images/IMG_6721.JPG','test_images/IMG_6722.JPG','test_images/IMG_6723.JPG','test_images/IMG_6724.JPG','test_images/IMG_6725.JPG','test_images/IMG_6726.JPG','test_images/IMG_6727.JPG']
    main(images)