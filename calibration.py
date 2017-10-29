import numpy as np
import cv2
import glob
import pickle

nx = 9
ny = 6
#Prepare object points, like ( 0,0,0), (1,0,0), (2,0,0)...(8,5,0) #9X6
objp = np.zeros((nx * ny,3), np.float32)
objp[:,:2]=np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

#Array to store object points and image points from all the images
objpoints = [] #3D points of real world space
imgpoints = [] #2D points of image plane

#Make list of calibration images
images = glob.glob(".\camera_cal\cal*.jpg")

#Iterate through the list and search of calibration corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Find checkerboard function
    ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)

    if ret == True:
        print("working on ", fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        #Draw and display corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        write_name = 'corners_found'+ str(idx)+'.jpg'
        cv2.imwrite(write_name, img)


#load image for reference
img=cv2.imread('.\camera_cal\calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

#Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

#Save Camera calibration image to later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("./calibration_pickle.p", "wb"))


