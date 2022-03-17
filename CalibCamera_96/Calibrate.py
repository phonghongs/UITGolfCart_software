import glob
import cv2  
import numpy as np

def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def calibration(cal_image_loc):
    '''
    Perform camera calibration.
    '''
    # Load in the chessboard calibration images to a list
    calibration_images = []

    for fname in cal_image_loc:
        img = cv2.imread(fname)
        calibration_images.append(img)

    # Prepare object points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays for later storing object points and image points
    objpoints = []
    imgpoints = []

    # Iterate through images for their points
    for image in calibration_images:
        ret, corners = cv2.findChessboardCorners(image, (9,6), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(image, (9, 6), corners, ret)
            cv2.imshow("IMGCAL", image)
            cv2.waitKey(0)
    # Returns camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                       (640, 480), None, 
                                                       None)
    return mtx, dist


cal_image_loc = glob.glob('*.jpg')
mtx, dist = calibration(cal_image_loc)

save_coefficients(mtx, dist, "calibration_chessboard_96deg.yml")