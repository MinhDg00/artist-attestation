import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm


## Image Processor Class for our Dataset
class ImageProcessor(Dataset):

    def __init__(self, path, metadata_df, mode = 'RGB', is_train = True, crop_width = 224, crop_height = 224, is_phone = False):
        self.path = path
        self.mode = mode
        self.metadata_df = metadata_df
        self.X,self.y = self.read_input(self.path)
        self.is_train = is_train
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.transforms = None
        if self.is_train:
            self.transforms = T.Compose([ T.ToTensor(),T.Normalize(mean = [0,0,0], std = [1,1,1]), T.RandomCrop(224), T.RandomHorizontalFlip()])
        else:
            if is_phone: 
                # self.transforms = T.Compose([T.ToTensor(),T.Normalize(mean = [0,0,0], std = [1,1,1]), T.Resize((1456, 992)), T.CenterCrop(224)]) 
                # self.transforms = T.Compose([T.ToTensor(),T.Normalize(mean = [0,0,0], std = [1,1,1]), T.Resize((224, 224))])
                self.transforms = T.Compose([T.ToTensor(),T.Normalize(mean = [0,0,0], std = [1,1,1]), T.Resize((512, 512)), T.CenterCrop(224)]) 
            else:
                self.transforms = T.Compose([T.ToTensor(),T.Normalize(mean = [0,0,0], std = [1,1,1]), T.CenterCrop(224)])
        

    # A simple function to get the file paths of all the training images and their corresponding labels
    def read_input(self,path):

        if os.path.isfile(path):
            _, filename = os.path.split(path)
            label = self.metadata_df[self.metadata_df['new_filename'] == filename, 'artist_idx'].iloc[0]
            return [path],[label]

        elif os.path.isdir(path):
            paths = []
            labels = []
            for i, df in self.metadata_df.iterrows():
                image = df['new_filename']
                img_path = os.path.join(path,image)
                label = df['artist_idx']
                paths.append(img_path)
                labels.append(label)
            return paths,labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):

        # Fetch image path and label
        path = self.X[idx]
        label = self.y[idx]

        img = cv2.imread(path)

        # Convert to BGR if necessary
        if self.mode == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Preprocess image
        crop = self.transforms(img)
        
        return crop.to(device), torch.Tensor([int(label)]).to(device)
