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

def train(model,train_loader,epochs = 2, lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, k = 3, freeze_base = False):
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    if freeze_base == True:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas = (beta_1,beta_2))
    else:
        optimizer = optim.Adam(model.parameters(), lr = lr, betas = (beta_1,beta_2))
    
    epoch_losses = []
    epoch_accuracies = []
    epoch_topk_accs = []
    
    for e in range(epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        correct = 0.0
        correct_k = 0.0
        total_size = 0
        acc = 0.0
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float().to(device))
            
            loss = loss_fn(outputs, labels.squeeze().type(torch.LongTensor).to(device))
            loss.backward()
            optimizer.step()
            
            batch_size = inputs.size(0)
                
            predicted = torch.topk(outputs,1,1).indices
            predicted_k = torch.topk(outputs,k,1).indices
            correct += (predicted == labels).sum().item()
            total_size += 1

            for j in range (batch_size):
                if labels[j,0] in predicted_k[j]:
                    correct_k += 1

            # print statistics
            running_loss += loss.item()
            print(f'[{e + 1}, {i * batch_size:5d}] batch_loss: {loss.item():.3f}')
        
        epoch_loss = (running_loss/total_size)
        acc = (correct/len(train_dataset)) * 100
        topk_acc = (correct_k/len(train_dataset))*100
        
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(acc)
        epoch_topk_accs.append(topk_acc)
        print(f'[Epoch {e + 1}: epoch_loss: {epoch_loss:.3f} accuracy: {acc:.3f} top-3_accuracy: {topk_acc:.3f}]')

    print('Finished Training')
    
    return {"loss": epoch_losses, "acc": epoch_accuracies, "topk_acc": epoch_topk_accs}