#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn

from typing import Tuple
from typing import List
from copy import deepcopy, copy

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST

from torchvision.transforms import Lambda
from torchvision.transforms import Compose, ToTensor

from torch import nn
from torch.optim import SGD
from torch.nn.functional import cross_entropy


from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(255/256, 1, N)    ## red  255,192, 203
vals[:, 1] = np.linspace(192/256, 1, N)
vals[:, 2] = np.linspace(203/256, 1, N)  ## blue 75, 0, 130
Pinks = ListedColormap(vals)


top = cm.get_cmap(Pinks, 256)
bottom = cm.get_cmap('Purples', 256)

newcolors = np.vstack((top(np.linspace(0, 1, 256)),
                       bottom(np.linspace(0, 1, 256))))

newcmp = ListedColormap(newcolors, name='PinkPueple')

############################################################################################################
############################################################################################################






############################################################################################################
############################################################################################################

def calculate_mean_and_std():
    train_data = MNIST(
        root='.',
        download=True,
        train=True, 
    )

    train_data.data = (train_data.data/255.0)      
    
    mean = train_data.data.mean()
    std = train_data.data.std()
    return mean, std



def download_normalized_data(mean, std):    
    
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x.view(784)),
            transforms.Lambda(lambda x: (x-mean)/std),       
    ]) 

    train_data = MNIST(
        root = '.', 
        download = True, 
        train = True, 
        transform = transform
    )

    test_data = MNIST(
        root='.',
        download=True,
        train=False, 
        transform = transform
    )
    
    return train_data, test_data





def calculate_mean_and_std_FMNIST():
    train_data = FashionMNIST(
        root='.',
        download=True,
        train=True,
    )

    train_data.data = (train_data.data/255.0)      
   
    mean = train_data.data.mean()
    std = train_data.data.std()
    return mean, std



def download_normalized_data_FMNIST(mean, std):    
   
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(784)),
            transforms.Lambda(lambda x: (x-mean)/std),      
    ])

    train_data = FashionMNIST(
        root = '.',
        download = True,
        train = True,
        transform = transform
    )

    test_data = FashionMNIST(
        root='.',
        download=True,
        train=False,
        transform = transform
    )
   
    return train_data, test_data





loss_tab = []
test_accuracy_tab = []
train_accuracy_tab = []

w1_mean_tab =[]
w1_std_tab =[]
w2_mean_tab =[]
w2_std_tab =[]
w3_mean_tab =[]
w3_std_tab =[]
w4_mean_tab =[]
w4_std_tab =[]
w5_mean_tab =[]
w5_std_tab =[]


b1_mean_tab =[]
b1_std_tab =[]
b2_mean_tab =[]
b2_std_tab =[]
b3_mean_tab =[]
b3_std_tab =[]
b4_mean_tab =[]
b4_std_tab =[]
b5_mean_tab =[]
b5_std_tab =[]


w1_tab =[]
w2_tab =[]
w3_tab =[]
w4_tab =[]
w5_tab =[]

b1_tab =[]
b2_tab =[]
b3_tab =[] 
b4_tab =[]
b5_tab =[]


