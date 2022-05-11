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
from torchvision.utils import make_grid


def visualize_layer1_filter(model):

    kernels = model.conv1.weight.detach().clone()
    print(kernels.size())
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    filter_img = make_grid(kernels, nrow = 8)
    plt.imshow(filter_img.cpu().permute(1, 2, 0))

def plot_conf_matrix(y_true,y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix)
    plt.figure(figsize=[15,20])
    disp.plot()
    plt.xticks(rotation = 45)
    plt.show()

def predict(model,x,k):
    
    # x is of shape (batch_size,C,H,W)
    # k: top k predicted result 
    
    output = model(x)    # (batchsize, numclass) 
    predicted_1 = torch.topk(output,1,1).indices
    predicted_k = torch.topk(output,k,1).indices
    return predicted_1.to(device), predicted_k.to(device)

def run_test(model, test_loader, k = 3):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    d = {} 
    correct = 0.0
    correct_k = 0.0
    total_size = 0
    acc = 0.0
    y_true = torch.Tensor().to(device)
    y_pred = torch.Tensor().to(device)
    test_samples = 0
    
    with torch.no_grad():
        for i,data in enumerate(test_loader,0):
             
            X, y = data
            pred, pred_k = predict(model, X, k)

            correct += (pred == y).sum().item()

            batch_size = X.size(0)
            for j in range (batch_size):
                if y[j,0] in pred_k[j]:
                    correct_k += 1

            y_true = torch.concat([y_true,y]).to(device)
            y_pred = torch.concat([y_pred,pred]).to(device)

            total_size += 1
            test_samples += X.size(0)
        
        
        d['acc'] = (correct/test_samples) * 100
        d['topk_acc'] = (correct_k/test_samples)*100
        
        p,r,fscore,_ = precision_recall_fscore_support(torch.flatten(y_true.cpu()), torch.flatten(y_pred.cpu()), average = 'weighted')
        d['p'], d['r'], d['fscore'] = p, r, fscore
        d['y_true'] = torch.flatten(y_true.cpu())
        d['y_pred'] = torch.flatten(y_pred.cpu())
        
    return d 