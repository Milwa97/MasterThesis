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
from torchvision.datasets import MNIST, FashionMNIST
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


############################################################################################################
############################################################################################################


class CustomNetwork(object):
    
    """
    Simple D-layer linear neural network 
    hidden_dims = topule(n0, n1, n2, ...nD)
    n0 = input layer
    n_D = output layer
    """
    
    def __init__(self, D, layers_dim):
        
        """
        Initialize network's weights according to Gaussian iid and network's biases with 0.0 values
        """
        
        self.weights = []
        self.biases = []
        
        self.D = len(layers_dim)-1
        assert self.D == D
        
        print("Depth of the network = number of hidden layers + 1:", D)
        
        for i in range(self.D):
            
            weight: torch.Tensor = torch.randn((layers_dim[i+1], layers_dim[i])) 
            bias: torch.Tensor = torch.zeros(layers_dim[i+1])  
            
            weight.requires_grad = True
            bias.requires_grad = True
            
            self.weights.append(weight)
            self.biases.append(bias)
       
            
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        """
                
        fl = (torch.matmul(x, self.weights[0].T) + self.biases[0])       

        for i in range(1,self.D):            
            out = ( torch.matmul(fl, self.weights[i].T) + self.biases[i])
            fl = out
        return torch.tanh(out)
    
    
    def parameters(self) -> List[torch.Tensor]:
        """
        Returns all trainable parameters 
        """
        return self.weights+self.biases




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


