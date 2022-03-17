# import cv2  
# import numpy as np

# #trackbar callback fucntion does nothing but required for trackbar
# def nothing(x):
# 	pass

# ROW = 220


# #create a seperate window named 'controls' for trackbar
# cv2.namedWindow('controls')
# #create trackbar in 'controls' window with name 'r''
# cv2.createTrackbar('r','controls',0,320,nothing)


# #create a while loop act as refresh for the view 
# while(1):
 
# 	#create a black image
#     img = cv2.imread('235.jpg')

# 	#returns current position/value of trackbar 
#     radius= int(cv2.getTrackbarPos('r','controls'))
# 	#draw a red circle in the center of the image with radius set by trackbar position
#     cv2.circle(img,(radius, ROW), 5, (0,0,255), -1)
#     cv2.circle(img,(640 - radius, ROW), 5, (0,0,255), -1)
# 	#show the image window
#     cv2.imshow('img',img)
	
# 	# waitfor the user to press escape and break the while loop 
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
		
# #destroys all window
# cv2.destroyAllWindows()


#______________________________________________________________________


import cv2
import numpy as np

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

mtx2, dist2 = load_coefficients('CalibCamera_96/calibration_chessboard_96deg.yml')

ROW = 480
img = cv2.imread('235.jpg')

H = 480
W = 640
H1 = 80
W1 = 180
H2 = 480
W2 = -293

src = np.float32([[W1, H1], [W - W1, H1], [W - W2, H2], [W2, H2]])
dst = np.float32([[0, 0], [640, 0], [640, ROW], [0, ROW]])

# compute IPM matrix and apply it
ipm_matrix = cv2.getPerspectiveTransform(src, dst)

cap = cv2.VideoCapture('test2.mp4')

while cap.isOpened():
    ret, img = cap.read()
    if ret:
        
        img = cv2.resize(img, (640, 480))
        ipm = cv2.warpPerspective(img, ipm_matrix, img.shape[:2][::-1])

        # display (or save) images
        cv2.imshow('img', img)
        cv2.imshow('ipm', ipm)
        
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()

#______________________________________________________________________

# import numpy as np
# import cv2

# img = cv2.imread('235.jpg')

# W2 = 292

# M = np.float32([
# 	[1, 0, W2],
# 	[0, 1, 0]
# ])

# imgH, imgW, _ = img.shape

# image = np.zeros((480, 640 + 2*W2, 3), dtype = "uint8")

# image[:, 0:imgW] = img

# shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
# cv2.imshow("myIMG", shifted)
# cv2.waitKey()