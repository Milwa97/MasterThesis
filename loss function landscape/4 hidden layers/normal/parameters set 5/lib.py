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
        
        fl = torch.tanh( torch.matmul(x, self.weights[0].T) + self.biases[0])      
        
        for i in range(1,self.D):            
            out = torch.tanh( torch.matmul(fl, self.weights[i].T) + self.biases[i])
            fl = out
        return out
    
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


############################################################################################################
############################################################################################################


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
