from PIL import Image
from torch.autograd import Variable

import glob
import torch
import torchvision

import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import scipy.misc
import timeit
import piexif

import math



array = np.loadtxt(open("values.csv", "rb"), delimiter=",", skiprows=1)
values = np.zeros(len(array))

for i in range(len(array)): 
    values[i] = int(np.argmax(array[i][1:]) + 1)
    

np.savetxt("output.csv", values, delimiter=",")