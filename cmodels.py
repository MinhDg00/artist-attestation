import pandas as pd
import glob
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.multiprocessing import Pool, Process, set_start_method
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Our Baseline Convolutional Neural Network
class BaselineConvnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, stride = 2, padding = 1)
        nn.init.zeros_(self.conv1.bias) 
        nn.init.normal_(self.conv1.weight, mean = 0, std = (math.sqrt(2/(32*3*3))))
        
        self.pool = nn.MaxPool2d(2)
        self.batch_norm2d = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        nn.init.zeros_(self.conv2.bias) 
        nn.init.normal_(self.conv2.weight, mean = 0, std = (math.sqrt(2/(32*3*3))))
        
        self.batch_norm1d = nn.BatchNorm1d(228)
        self.fc1 = nn.Linear(6272, 228)
        self.fc2 = nn.Linear(228, 57)

    def forward(self, x):
        
        # Convolution stack 1
        x = self.conv1(x)
        x = self.batch_norm2d(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Convolution stack 2
        x = self.conv2(x)
        x = self.batch_norm2d(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten and FC layers
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.batch_norm1d(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

# Loads desired pre-built model
def loadPretrainedModels(name,num_classes,pretrained = True, freeze_base = False):
    
    if name == 'resnet-18':
        model = models.resnet18(pretrained)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat,num_classes)
        if freeze_base == True:
            for param in model.parameters():
                param.requires_grad = False
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        return model

# Unfreeze base layers for training
def unfreeze_base(model):
    m = model
    for param in m.parameters():
        param.requires_grad = True
    return m