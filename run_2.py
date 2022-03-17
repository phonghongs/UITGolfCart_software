import cv2
import numpy as np
import threading
import os

from simple_pid import PID
from Laneline.net import Net
from Laneline.src.util import calcul_speed
from Laneline.src.parameters import Parameters
from Laneline.utils.connect import Connection
from Laneline.utils.controller import Controller
from Libs.laneprocess import CenterCalv2
from Libs.utils import SimpleKalmanFilter, load_coefficients
from Libs.car import Car
from Libs.config import args_setting
from Libs import config
import matplotlib.pyplot as plt
import math

import torch
from torch.utils.data.dataloader import DataLoader
import albumentations as A
from RoadSegment.ex import BiSeNet
from RoadSegment.dataset import UTEDataset
from RoadSegment.helpers import (load_checkpoint, 
                    reverse_one_hot,
                    colour_code_segmentation,
                    img_show,
                    to_rgb) 
from PIL import Image
from albumentations.pytorch import ToTensorV2

args = args_setting()

global DEBUG, CENTER_LINE
DEBUG = True
MAX_SPEED = 80
MAX_ANGLE = 25
CENTER_LINE = 220
DEVICE = "cuda"

font = cv2.FONT_HERSHEY_SIMPLEX
golfCart = Car(MAX_SPEED, args.serialtest)

model = BiSeNet(4)
load_checkpoint(torch.load('./RoadSegment/checkpoints/best_model.pth'), model)

test_transform = A.Compose(
    [
        A.Resize(height=480, width=600),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            # max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

model = model.cuda()
model.eval()


if __name__ == "__main__":
    net = Net()
    p = Parameters()
    net.load_model(config.weight)

    km = SimpleKalmanFilter(1, 5, 5)

    kp = 0.7
    ki = 0
    kd = 0.1
    pid = PID(kp, ki, kd, setpoint= 0)
    pid.output_limits = (-25, 25)

    cap = cv2.VideoCapture("test.mp4")
    mtx2, dist2 = load_coefficients('CalibCamera_96/calibration_chessboard_96deg.yml')

    print("We are currently using Manual mode")

    while cap.isOpened() and not golfCart.done:  
        ret, frame = cap.read()
        if ret == True:
            image_resized = cv2.resize(frame[50:], (512, 256))
            image_resized = cv2.undistort(image_resized, mtx2, dist2, None, None)
            img4lane = np.copy(image_resized)
            img4segment = np.copy(image_resized)

            y , x = net.predict(img4lane, warp = False)    

            # #Segment
            # augmentations = test_transform(image=img4segment)
            # input = augmentations["image"] 
            # input = input.unsqueeze(0)
            # input = input.cuda()
            # output = model(input)
            # output = torch.argmax(output,1)[0]
            # # # print(output.size())
            # # ##reverse mapping
            # mapping = {
            #     (31,120,180):0,
            #     (227,26,28) :1,
            #     (106,61,154):2,
            #     (0, 0, 0)   :3,
            #     }
            # rev_mapping = {mapping[k]: k for k in mapping}

            # pred_image = torch.zeros(3, output.size(0), output.size(1), dtype=torch.uint8)
            # # print('pred_image',pred_image.size())
            # for k in rev_mapping:
            #     pred_image[:, output==k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)
            # final_img = pred_image.permute(1, 2, 0).numpy()
            # final_img = cv2.resize(final_img, (512, 256))

            point, image_points_center = CenterCalv2(x, y, image_resized, CENTER_LINE)

            angle = math.atan((point[1] - 256)/(256-CENTER_LINE))*180/math.pi
            angle = pid(-(angle*0.3))

            speed = int(calcul_speed(angle, MAX_SPEED, MAX_ANGLE)) 
            speed_ratio = speed/MAX_SPEED
            predicted_speed = speed_ratio * 5 + 56


            if (DEBUG):
                image_points_result = net.get_image_points()
                image_points_center = cv2.putText(image_points_center, f'{angle} - {predicted_speed}', (50, 50), font, 
                   1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("image_points_center", image_points_center)
                # cv2.imshow("final segment", final_img)
                cv2.imshow("image_points_result", image_points_result)
    
            if  golfCart.auto == True:
                print("predicted_speed, angle: ", predicted_speed, angle)
                golfCart.RunAuto(predicted_speed, angle)

            cv2.waitKey(1) 

    if DEBUG:
        cv2.destroyAllWindows()
