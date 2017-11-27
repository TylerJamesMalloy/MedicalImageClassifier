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

start = timeit.default_timer()

### 32x32 CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""
# CNN Model (2 conv layer)
class Net(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
"""

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('Test')

# Iterate through Type 1 image files 
running_loss = 0.0

feature_list = []
target_list = []

for type in classes:
    image_folder = glob.iglob("../processed_images/Full_Size/" + type + "/*.jpg")
    for filename in image_folder:
        piexif.remove(filename)
        image = Image.open(filename)
        # 32x32 now just for testing, need to figure out best dimensions
        try:
            image = scipy.misc.imresize(image, (32, 32))
        except ValueError:
            continue 
        image = np.array(image)
        image = np.swapaxes(image,0,2)
        feature_list.append(image)
        target_list.append(i)

feature_array = np.array(feature_list)
features = torch.from_numpy(feature_array)

target_array = np.array(target_list)
targets = torch.from_numpy(target_array)
targets = targets

train = TensorDataset(features, targets)
trainloader = DataLoader(train, batch_size=targets.shape[0], shuffle=False)


net = Net()
net.load_state_dict(torch.load('Neural_Networks/Traditional_CNN.pth'))

error = 0.0
running_loss = 0.0
running_total = 0.0
for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs, labels = data

    # wrap them in Variable
    inputs, labels = Variable(inputs), Variable(labels)
    labels = labels.long()
    inputs = inputs.float()
    numpyout = net(inputs).data.numpy()
    outputs = np.zeros(numpyout.shape[0])
    for i in range(labels.size()[0]):
        outlist = numpyout[i].tolist()
        outputs[i] = outlist.index(max(outlist))

    inputs = labels.data.numpy()
    running_loss += sum(abs(inputs - outputs))
    running_total += numpyout.shape[0]

error = running_loss / running_total
print(error)