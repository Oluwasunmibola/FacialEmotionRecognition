import os
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt

class FER2013Dataset(Dataset):

    def __init__(self, csv_file, img_dir,  transform=None):
        '''
        FER2013Dataset
        :param csv_file:  path to csv file
        :param img_dir: pathe to Images
        :param transform: tranform to be applied to the dataset
        '''
        # load data: initializing parameters
        self.img_dir = img_dir
        self.csv_file = pd.read_csv(csv_file)
        self.classes = self.csv_file['emotion']
        self.transform = transform

    def __getitem__(self, idx):
        # accessing data through index
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # reading the image
        image = Image.open(self.img_dir+str(idx)+'.jpg')
        # reading classes by index which is in numpy
        classes = np.array(self.classes[idx])
        # converting the labels from numpy to tensor
        classes = torch.from_numpy(classes)

        if self.transform:
            img = self.transform(image)
        return img, classes

    def __len__(self):
        return len(self.csv_file)








