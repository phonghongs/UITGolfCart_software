import os
from numpy.core.fromnumeric import searchsorted
import torch
from torch.utils import data
from tqdm import tqdm
from torchvision import transforms
from collections import OrderedDict
from torch.utils.data import Dataset
from ex import BiSeNet
import numpy as np
from PIL import Image
from glob import glob

class UTEDataset(Dataset):
    # color_encoding = [
    #     ('road', (31,120,180)),
    #     ('unlabeled', (0,0,0)),
    #     ]
    color_encoding = [
        ('road', (31,120,180)),
        ('people', (227,26,28)),
        ('car', (106,61,154)),
        ('unlabeled', (0, 0, 0))
        ]

    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        #Normalization
        self.normalize = transforms.Compose([
            #transforms.Resize((360,640)),
            transforms.Resize((480,480)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ]) ##imagenet norm

        self.DATA_PATH = os.path.join(os.getcwd(), 'dataset/UTE_Dataset/')
        self.train_path = os.path.join(self.DATA_PATH, 'image')

        self.data_files = self.get_files(self.train_path)
        self.label_files = [self.get_label_file(f, 'image', 'label') for f in self.data_files]

    def __len__(self):
        print(len(self.data_files))
        return len(self.data_files)

    def get_files(self, data_folder):
        #
        return  glob("{}/*.{}".format(data_folder, 'png'))

    def get_label_file(self, data_path, data_dir, label_dir):
        #
        data_path = data_path.replace(data_dir, label_dir)
        print(data_path)
        frame, ext = data_path.split('.')
        return "{}.{}".format(frame, ext)

    def image_loader(self, data_path, label_path):
        data = Image.open(data_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        return data, label.resize((480,480))

    def label_decode_cross_entropy(self, label):
        """
        Convert label image to matrix classes for apply cross entropy loss. 
        Return semantic index, label in enumemap of H x W x class
        """
        semantic_map = np.zeros(label.shape[:-1])
        #Fill all value with 0 - defaul
        semantic_map.fill(self.num_classes - 1) #self.num_classes - 1
        #Fill the pixel with correct class
        for class_index, color_info in enumerate(self.color_encoding):
            color = np.array(color_info[1])
            # print(color.shape)
            # print(label.shape)
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = class_index
            # print(semantic_map)
        return semantic_map

    def __getitem__(self, index):
        """
            Args:
            - index (``int``): index of the item in the dataset
            Returns:
            A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
            of the image.
        """
        data_path, label_path = self.data_files[index], self.label_files[index]
        img, label = self.image_loader(data_path, label_path)
        # img.show()
        # label.show()
        # Normalize image
        img = self.normalize(img)
        # Convert label for cross entropy
        label = np.array(label)
        label = self.label_decode_cross_entropy(label)
        # print(label)
        label = torch.from_numpy(label).long()

        return img, label


        
    