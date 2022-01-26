from distutils.command.config import config


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset
import yaml
print(f"pyTorch version {torch.__version__}")
print(f"torchvision version {torchvision.__version__}")
print(f"CUDA available {torch.cuda.is_available()}")

config = None
with open('config.yml') as f:
    config = yaml.safe_load(f)

   
    
n1 = config['net']['n1']
n2 = config['net']['n2']
n3 = config['net']['n3']

class CustomNet(torch.nn.Module):
    def __init__(self, input_nc, n1, n2, n3, n_classes):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, n1, (5, 5), padding='valid')
        self.conv2 = nn.Conv2d(n1, n2, (5, 5), padding='valid')
        self.linear1 = nn.Linear(5408, config["train"]["bs"]) # 4*4 image dimension after 2 max_pooling
        self.linear2 = nn.Linear(config["train"]["bs"], n_classes)
        self.max_pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x