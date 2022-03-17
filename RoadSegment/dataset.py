import os
from numpy.core.fromnumeric import searchsorted
import torch
from torch.utils import data
from tqdm import tqdm
from torchvision import transforms
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from RoadSegment.ex import BiSeNet
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
        ('unlabeled', (0,0,0)),
        ]
    # color_encoding = [
    #     ('sky', (128, 128, 128)),
    #     ('building', (128, 0, 0)),
    #     ('pole', (192, 192, 128)),
    #     ('road_marking', (255, 69, 0)),
    #     ('road', (128, 64, 128)),
    #     ('pavement', (60, 40, 222)),
    #     ('tree', (128, 128, 0)),
    #     ('sign_symbol', (192, 128, 128)),
    #     ('fence', (64, 64, 128)),
    #     ('car', (64, 0, 128)),
    #     ('pedestrian', (64, 64, 0)),
    #     ('bicyclist', (0, 128, 192)),
    #     ('unlabeled', (0, 0, 0))
    # ]

    def __init__(self, mode='train', num_classes=4):
        self.mode = mode
        self.num_classes = num_classes
        #Normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ]) ##imagenet norm

        self.DATA_PATH = os.path.join(os.getcwd(), 'dataset\\')
        self.train_path, self.val_path, self.test_path = [os.path.join(self.DATA_PATH, x) for x in ['image','val','test']]

        if self.mode == 'train':
            self.data_files = self.get_files(self.train_path)
            self.label_files = [self.get_label_file(f, 'image', 'labeltrain') for f in self.data_files]
        elif self.mode == 'val':
            self.data_files = self.get_files(self.val_path)
            self.label_files = [self.get_label_file(f, 'val', 'val_label') for f in self.data_files]
        elif self.mode == 'test':
            self.data_files = self.get_files(self.test_path)
            self.label_files = [self.get_label_file(f, 'test', 'test_label') for f in self.data_files]
        else: 
            raise RuntimeError("Unexpected dataset mode."
                                "Supported modes: train, val, test")

    def __len__(self):
        return len(self.data_files)

    def get_files(self, data_folder):
        #
        return  glob("{}/*.{}".format(data_folder, 'jpg'))

    def get_label_file(self, data_path, data_dir, label_dir):
        #
        data_path = data_path.replace(data_dir, label_dir)
        frame, ext = data_path.split('.')
        print(frame, ext)
        return "{}.{}".format(frame, 'png')

    def image_loader(self, data_path, label_path):
        data = Image.open(data_path)
        label = Image.open(label_path)
        return data.resize((480,600)), label.resize((480,600))

    def label_decode_cross_entropy(self, label):
        """
        Convert label image to matrix classes for apply cross entropy loss. 
        Return semantic index, label in enumemap of H x W x class
        """
        semantic_map = np.zeros(label.shape[:-1])
        #Fill all value with 0 - defaul
        semantic_map.fill(0)
        #Fill the pixel with correct class
        for class_index, color_info in enumerate(self.color_encoding):
            color = color_info[1]
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = class_index
            #print(semantic_map)
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
        
        # Normalize image
        img = self.normalize(img)
        # Convert label for cross entropy
        label = np.array(label)
        label = self.label_decode_cross_entropy(label)
        # print(label)
        label = torch.from_numpy(label).long()

        return img, label



        
    
