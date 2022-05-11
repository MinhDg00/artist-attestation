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

def merge_result(*argv):
    d = defaultdict(list)
    for arg in argv:
        if isinstance(arg, dict):
            for k in arg.keys(): 
                if isinstance(k, list):
                    d[k] += arg[k]
                else:
                    d[k].append(arg[k])
    
    return d    

def save_result(tbl, path):
    f = open(path, 'w')
    for k in tbl:
        values = ' '.join([str(n) for n in tbl[k]])
        f.write(values + "\n")
    f.close()    


def get_model_metrics(path):
    f = open(path, 'r')
    losses, accs, topk_accs = f.readlines()
    losses, accs, topk_accs = losses.strip().split(' '), accs.strip().split(' '), topk_accs.strip().split(' ')
    losses, accs, topk_accs = list(map(float, losses)), list(map(float, accs)), list(map(float, topk_accs))
    
    return {'loss': losses, 'acc': accs, 'topk_acc': topk_accs} 

def find_corrupt_imgs(df, path, transform): 
    invalids = set()
    for i, df in tqdm(df.iterrows()):
        image = df['new_filename']
        img_path = os.path.join(path,image)
        img = cv2.imread(img_path)
        try:
            img = transform(img)
        except TypeError:
            invalids.add(image)
            
    return invalids

def remove_corrupt_imgs(metadata_df, corrupt_list):
    for img in corrupt_list:
        metadata_df = metadata_df[metadata_df['new_filename'] != img]
    return metadata_df