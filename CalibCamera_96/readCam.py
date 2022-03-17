import cv2
import numpy as np
from threading import Thread, Lock

dispW=640
dispH=480
flip=2

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

cap2 = cv2.VideoCapture(1)
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

i = 0

while True:
    _, img0 = cap2.read()

    cv2.imshow("IMG0", img0)
    
    k =cv2.waitKey(1)

    if k == 27:
        break
    if k == ord('c'):
        i+= 1
        cv2.imwrite(f'{i}.jpg', img0)

cap2.release()