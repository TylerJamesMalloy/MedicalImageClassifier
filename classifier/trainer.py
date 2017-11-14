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

torch.save(net,'UntrainedNN.pt')

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

for i in range(0,2):
    for filename in glob.iglob("../train/Type_" + str(i + 1) + "/*.jpg"):
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
print(targets.shape)

train = TensorDataset(features, targets)
trainloader = DataLoader(train, batch_size=50, shuffle=True)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

torch.save(net,'TrainedNN.pt')

"""

dataiter = iter(trainloader)
images, labels = dataiter.next()

images = images.float()
labels = labels.long()

outputs = net(Variable(images))

_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(10)))
"""

stop = timeit.default_timer()

print(stop - start)