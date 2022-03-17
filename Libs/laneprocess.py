from Libs.utils import warpPers
from scipy import stats
import numpy as np
import cv2

def CenterCalv2(x, y, image = [], sample = 200, BEV = True):
    xBEV = []
    yBEV = []

    slope4Single = 0
    intercept4Single = 0
    slope_1 = 0
    slope_2 = 0
    intercept_1 = 0
    intercept_2 = 0

    if not BEV:
        xBEV = x
        yBEV = y
    else:

        src = np.float32([[0, 256], [512, 256], [0, 0], [512, 0]])
        dst = np.float32([[200, 256], [312, 256], [0, 0], [512, 0]])

        M = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(image, M, (512, 256))
        # print(M)
        # print("_________________")

        for i in range(0, len(x)):
            cacheX = []
            cacheY = []
            for index, point in enumerate(x[i]):
                yCh, xCh = warpPers(y[i][index], point, M)
                image = cv2.circle(image, (int(yCh), int(xCh)), 5, [255, 255, 0], -1)
                cacheX.append(int(xCh))
                cacheY.append(int(yCh))

            xBEV.append(cacheX)
            yBEV.append(cacheY)
 
    if (len(xBEV) == 1):
        x_p1, y_p1 = np.array(xBEV[0]), np.array(yBEV[0])
        slope_1, intercept_1, r, p, std_err = stats.linregress(x_p1, y_p1)
        slope_2, intercept_2 = 0, intercept_1 + 60

    if (len(xBEV) == 2):
        x_p1, y_p1 = np.array(xBEV[0]), np.array(yBEV[0])
        slope_1, intercept_1, r, p, std_err = stats.linregress(x_p1, y_p1)

        x_p2, y_p2 = np.array(xBEV[1]), np.array(yBEV[1])
        slope_2, intercept_2, r, p, std_err = stats.linregress(x_p2, y_p2)

    y1 = slope_1*sample + intercept_1
    y2 = slope_2*sample + intercept_2
    ySample =  y2 + (y1 - y2)/2

    image = cv2.circle(image, (int(ySample), sample), 5, [255, 0, 0], -1)

    return (sample , ySample), image
