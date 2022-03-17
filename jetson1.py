
from __future__ import division
from pickle import TRUE
import cv2
import numpy as np
import socket
import struct
from Libs.config import args_setting
from Libs import config
from Libs.laneprocess import CenterCalv2
from Libs.utils import SimpleKalmanFilter, load_coefficients
from Laneline.net import Net
import math
import time

CENTER_LINE = 220
CLIENT_ID = "JETSON1"
MAX_DGRAM = 2**16
MAX_IMAGE_DGRAM = MAX_DGRAM - 64 # extract 64 bytes in case UDP frame overflown

def main():
    """ Getting image udp frame &
    concate before decode and output image """

    # Set up socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 15555))

    cap = cv2.VideoCapture('test.mp4')

    net = Net()
    net.load_model(config.weight)
    mtx2, dist2 = load_coefficients('CalibCamera_96/calibration_chessboard_96deg.yml')
    done = False

    while not done and cap.isOpened():
        ret, img = cap.read()
        if ret:
            image_resized = cv2.resize(img[50:], (512, 256))
            image_resized = cv2.undistort(image_resized, mtx2, dist2, None, None)

            y , x = net.predict(image_resized, warp = False)
            point, image_points_center = CenterCalv2(x, y, image_resized, CENTER_LINE)

            # cv2.imshow("image_points_center", image_points_center)

            lanelineData = f'TX2_1:{str(x)};{str(y)}'
            s.send(lanelineData.encode('utf8'))
            seg = s.recv(MAX_DGRAM)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     done = True

    s.send("quit:".encode('utf8'))
    time.sleep(1)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
