from PIL import Image
from torch.autograd import Variable

import glob
import torch
import torchvision

import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

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

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

classes = ('Type_1','Type_2','Type_3')

squeeze = models.squeezenet1_1(pretrained=True)

# Iterate through Type 1 image files 
running_loss = 0.0

for filename in glob.iglob('../train/Type_1_small/*.jpg'):

    image = Image.open(filename)
    img_tensor = transform(image)
    img_tensor.unsqueeze_(0)

    input, label = Variable(img_tensor), Variable(torch.FloatTensor([1]))
    
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    output = net(input)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

"""

img_variable = Variable(img_tensor)
fc_out = squeeze(img_variable)
print(fc_out.data.numpy().argmax())

# Iterate through Type 2 image files
for filename in glob.iglob('../train/Type_2/*.jpg'):
    image = Image.open(filename)
    img_tensor = transform(image)
    img_tensor.unsqueeze_(0)

# Iterate through Type 3 image files
for filename in glob.iglob('../train/Type_3/*.jpg'):
    image = Image.open(filename)
    img_tensor = transform(image)
    img_tensor.unsqueeze_(0)
"""