import random
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, path, mode = 'RGB', is_train = True, crop_width = 224, crop_height = 224):
        self.path = path
        self.mode = mode
        self.dataset = self.read_input(self.path)
        self.sample_idx = 0
        self.is_train = is_train
        self.crop_width = crop_width
        self.crop_height = crop_height

    def read_input(self,path):
        if os.path.isfile(path):
            return [path]
        elif os.path.isdir(path):
            paths = [os.path.join(path,image) for image in os.listdir(path)]
            return paths

    def __iter__(self):
        self.sample_idx = 0
        return self

    def __len__(self):
        return len(self.dataset)

    def __next__(self):

        if self.sample_idx < len(self.dataset):
            
            # Read image data
            path = self.dataset[self.sample_idx]
            img = cv2.imread(path)

            # Convert to BGR if necessary
            if self.mode == 'BGR':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            crop = self.preprocess(img)

            self.sample_idx += 1
            return crop
        else:
            raise StopIteration

    def preprocess(self,img):
        
        image = img
        
        # STEP 1: Zero-Centering and Normalization

        # Normalize pixels
        image = image / 255

        # Zero center pixels across the image for each channel
        means = np.mean(image,axis = (0,1))
        image[:,:,0] = image[:,:,0] - means[0]
        image[:,:,1] = image[:,:,1] - means[1]
        image[:,:,2] = image[:,:,2] - means[2]

        # STEP 2: get crop of image
        
        cw = self.crop_width
        ch = self.crop_height
        
        # Cropping for training set
        if self.is_train:

            # Randomly flip image with a 50% chance
            choose = [0,1]
            hf = random.choices(choose, weights = [0.5,0.5], k = 1)
            if hf[0] == 1:
                image = cv2.flip(image , 1)

            # Get random (crop_width X crop_height) crop of image
            max_h = image.shape[0] - ch
            max_w = image.shape[1] - cw
            x = np.random.randint(0, max_w)
            y = np.random.randint(0, max_h)
            crop = image[y: y + ch, x: x + cw]
        
        # Cropping for test and validation set
        else:
            
            # Get a center crop of the image
            center = np.array(list(image.shape)) / 2
            mid_h = center[0] - ch/2
            mid_w = center[1] - cw/2
            crop = image[int(mid_h):int(mid_h + ch), int(mid_w):int(mid_w + cw)]

        return crop

test_img = 'batch_test/'

imobj = ImageProcessor(test_img)

for idx,i in enumerate(imobj):
    cv2.imshow(str(idx),i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
