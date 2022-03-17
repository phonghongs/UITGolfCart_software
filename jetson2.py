
from __future__ import division
from pickle import TRUE
from weakref import finalize
import cv2
from matplotlib.pyplot import contour
import numpy as np
import socket
import sys
import struct
import time
from Libs.config import args_setting
from Libs import config
from Libs.laneprocess import CenterCalv2
from Libs.utils import SimpleKalmanFilter, load_coefficients
import albumentations as A
from albumentations.pytorch import ToTensorV2
from RoadSegment.ex import BiSeNet
from RoadSegment.dataset import UTEDataset
from RoadSegment.helpers import (load_checkpoint, 
                    reverse_one_hot,
                    colour_code_segmentation,
                    img_show,
                    to_rgb) 

import torch
from torch.utils.data.dataloader import DataLoader

MAX_DGRAM = 2**16
CENTER_LINE = 220
CLIENT_ID = "JETSON2"
blank_image = np.zeros((256,512,3), np.uint8)

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


def main():
    """ Getting image udp frame &
    concate before decode and output image """

    # Set up socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 15555))
    done = False
    cap = cv2.VideoCapture('test.mp4')

    model = BiSeNet(4)
    load_checkpoint(torch.load('./RoadSegment/checkpoints/best_model.pth'), model)

    mtx2, dist2 = load_coefficients('CalibCamera_96/calibration_chessboard_96deg.yml')
    model = model.cuda()
    model.eval()

    kerel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    print("OK")
    while not done and cap.isOpened():
        ret, img = cap.read()
        if ret:
            image_resized = cv2.resize(img[50:], (512, 256))
            image_resized = cv2.undistort(image_resized, mtx2, dist2, None, None)
            augmentations = test_transform(image=image_resized)
            input = augmentations["image"] 
            input = input.unsqueeze(0)
            input = input.cuda()
            output = model(input)
            output = torch.argmax(output,1)[0]
            mapping = {
                255:0
            }
            rev_mapping = {mapping[k]: k for k in mapping}

            pred_image = torch.zeros(1, output.size(0), output.size(1), dtype=torch.uint8)
            for k in rev_mapping:
                pred_image[:, output==k] = torch.tensor(rev_mapping[k]).byte().view(1, 1)

            predict_img = pred_image.permute(1, 2, 0).numpy()
            predict_img = cv2.resize(predict_img, (512, 256))


            thre_mask = cv2.morphologyEx(predict_img,cv2.MORPH_DILATE,kerel5)

            conts, hier = cv2.findContours(thre_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cont = sorted(conts, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)[0].tolist()
            finalCont = [cnt[0] for cnt in cont]

            lanelineData = f'TX2_2:{str(finalCont)}'
            s.send(lanelineData.encode('utf8'))
            seg = s.recv(MAX_DGRAM)

            cv2.imshow("SEG", image_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                done = True

    s.send("quit".encode('utf8'))
    time.sleep(1)
    s.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
