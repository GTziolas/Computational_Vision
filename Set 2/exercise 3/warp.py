#2591

# =============================================================================
# The 4 points need to be clicked in a clockwise manner on the image
# starting from top left and ending at bottom left.
# Then press right click to complete the cropping and start
# the warping process.
# In the end press ESC to exit and terminate the program.
#
# Help for the calculation of the homography matrix was taken from:
# https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog 
# =============================================================================
import numpy as np
import cv2 as cv
import sys

#input image is argv[1] and output image is argv[2]
input_image = cv.imread(sys.argv[1])
points = []

def find_homography_matrix(p1, p2):
    A = []
    for i in range(0,len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
        
    A = np.asarray(A)
    U, S, VTrans = np.linalg.svd(A)
    #divide the VTrans vector with the last element of the VTrans vector
    #so we get 1 for h33 but it's not explicitly needed as we can get the
    #same result with the code commented below:
    L = VTrans[-1] / VTrans[-1,-1]  # Vlast = VTrans[-1]
    H = L.reshape(3, 3)             # Vlastreshape = Vlast.reshape(3,3)
    return H                        # return Vlastreshape 
    
def click_event(event, x, y, flags, param):
    global points
    if (event == cv.EVENT_LBUTTONDOWN):
        points.append((x,y))
        #paint the selected points but display them in the final image
        #remove the following line if you want
        cv.circle(img, (x,y), 8, (255,0,0),-1)
        cv.imshow('image', img)
    elif (event == cv.EVENT_RBUTTONDOWN):
        minx = min(points, key=lambda x: (x[0]))[0]
        maxx = max(points, key=lambda x: (x[0]))[0]
        miny = min(points, key=lambda x: (x[1]))[1]
        maxy = max(points, key=lambda x: (x[1]))[1]        
        print('(minx, maxx, miny, maxy):', minx, maxx, miny, maxy)
        
        size = (1000,1000,3)
        im_dst = np.zeros(size,np.uint8)
        #size[0] is the width, size[1] is the height, -1 is so we dont go out of image bounds
        destination_points = np.array([[0,0], [size[0]-1,0],[size[0]-1,size[1]-1],[0,size[1]-1]],dtype=float)
        source_points = np.array([points[0],points[1],points[2],points[3]])
        h = find_homography_matrix(source_points,destination_points)
        
        im_dst = cv.warpPerspective(img, h, (size[1], size[0])) #width,height
        cv.imshow('Warped', im_dst)
        cv.imwrite(sys.argv[2], im_dst)
        
        
img = cv.imread(sys.argv[1])
cv.imshow('image', input_image)

cv.setMouseCallback('image', click_event)
#Terminate with ESC
cv.waitKey(0)
cv.destroyAllWindows()

