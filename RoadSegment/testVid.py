import torch
import os
import cv2
import numpy as np
from torch.utils.data.dataloader import DataLoader
import albumentations as A
from ex import BiSeNet
from dataset import UTEDataset
from helpers import (load_checkpoint, 
                    reverse_one_hot,
                    colour_code_segmentation,
                    img_show,
                    to_rgb) 
from PIL import Image
from albumentations.pytorch import ToTensorV2



DEVICE = "cuda"
folder = 'results'
model = BiSeNet(4)
load_checkpoint(torch.load('checkpoints/best_model.pth'), model)

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

x = 0

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

cap = cv2.VideoCapture("test.mp4")

mtx2, dist2 = load_coefficients('CalibCamera_96/calibration_chessboard_96deg.yml')

while cap.isOpened():
    ret, img = cap.read()

    if ret:
        ##transform
        image_resized = cv2.resize(img[50:], (512, 256))
        image_resized = cv2.undistort(image_resized, mtx2, dist2, None, None)
        augmentations = test_transform(image=image_resized)

        input = augmentations["image"] 

        ##unsqueeze
        input = input.unsqueeze(0)

        ##evaluate

        ##prediction
        input = input.cuda()
        output = model(input)
        output = torch.argmax(output,1)[0]
        # # print(output.size())
        # ##reverse mapping
        mapping = {
            (31,120,180):0,
            (227,26,28) :1,
            (106,61,154):2,
            (0, 0, 0)   :3,
            }
        rev_mapping = {mapping[k]: k for k in mapping}

        pred_image = torch.zeros(3, output.size(0), output.size(1), dtype=torch.uint8)
        # print('pred_image',pred_image.size())
        for k in rev_mapping:
            pred_image[:, output==k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)
        final_img = pred_image.permute(1, 2, 0).numpy()
        final_img = cv2.resize(final_img, (512, 256))
        vis = np.concatenate((image_resized, final_img), axis=0)
        cv2.imshow("final_img", vis)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()