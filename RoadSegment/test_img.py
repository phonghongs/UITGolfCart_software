import torch
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader
import albumentations as A
from ex4 import BiSeNet
from dataset import UTEDataset
from helpers import (load_checkpoint, 
                    reverse_one_hot,
                    colour_code_segmentation,
                    img_show,
                    to_rgb) 
from PIL import Image
from albumentations.pytorch import ToTensorV2


DEVICE = "cuda"
folder = 'results' #THƯ MỤC LƯU DATA PREDICT NÈ
model = BiSeNet(4)
load_checkpoint(torch.load('checkpoints/best_model_ex4.pth'), model)

x = 0


# dataset_test = CamVidDataset(mode='test')
# dataloader_test = DataLoader(
#     dataset_test,
#     batch_size=1,
#     shuffle=True
# )


##load image
image_dir = 'car' # THƯ MỤC DATA TEST NÈ
images = os.listdir(image_dir)
with torch.no_grad():
    for idx in range(3):
        img_path = os.path.join(image_dir, images[idx])
        img = np.array(Image.open(img_path).convert("RGB"))

        ##transform
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
        augmentations = test_transform(image=img)
        input = augmentations["image"] 

        ##unsqueeze
        input = input.unsqueeze(0)

        ##evaluate
        model.eval()

        ##prediction
        output = model(input).to(device=DEVICE)
        output = torch.argmax(output,1)[0]
        print(output.size())
        ##reverse mapping
        mapping = {
            #(31,120,180)  :0 
            (31,120,180):0,
            (227,26,28) :1,
            (106,61,154):2,
            (0, 0, 0)   :3,
            }
        rev_mapping = {mapping[k]: k for k in mapping}

        pred_image = torch.zeros(3, output.size(0), output.size(1), dtype=torch.uint8)
        print('pred_image',pred_image.size())
        for k in rev_mapping:
            pred_image[:, output==k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)
        final_img = pred_image.permute(1, 2, 0).numpy()
        final_img = Image.fromarray(final_img,'RGB')
        print(x)
        final_img.save(f'{folder}/{images[idx]}_predict.png')
        x = x + 1
        
        final_img.show()
