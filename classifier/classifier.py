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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('Type_1','Type_2','Type_3')

# Iterate through Type 1 image files 
running_loss = 0.0

feature_list = []
target_list = []

for filename in glob.iglob('../train/Type_1_small/*.jpg'):
    image = Image.open(filename)
    # 320x240 now just for testing, need to figure out best dimensions
    image = scipy.misc.imresize(image, (320, 240))
    image = np.array(image)
    print(image.shape)
    feature_list.append(image)
    target_list.append('1')

feature_array = np.array(feature_list)
features = torch.from_numpy(feature_array)

target_array = np.array(feature_list)
targets = torch.from_numpy(target_array)

train = TensorDataset(features, targets)
train_loader = DataLoader(train, batch_size=50, shuffle=True)