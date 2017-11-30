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

start = timeit.default_timer()
# 256x256 CNN
# CNN Model (2 conv layer)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(131072, 3)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

net = Net().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('Type_1','Type_2','Type_3', 'AType_1','AType_2','AType_3')

# Iterate through Type 1 image files 
running_loss = 0.0

feature_list = []
target_list = []

# post-pre-processing-processing 
# Train data post-pre-processing-processing 
for type in classes:
    feature_id = 0
    if(type == "Type_1" or type == "AType_1"):
        feature_id = 1
    elif(type == "Type_2" or type == "AType_2"): 
        feature_id = 2
    elif(type == "Type_3" or type == "AType_3"):
        feature_id = 3
    else: 
        continue

    image_folder = glob.iglob("../processed_images/Full_Size/" + type + "/*.jpg")

    for filename in image_folder:
        piexif.remove(filename)
        image = Image.open(filename)
        try:
            image = scipy.misc.imresize(image, (256, 256))
        except ValueError:
            continue 
        image = np.array(image)
        image = np.swapaxes(image,0,2)
        feature_list.append(image)
        target_list.append(feature_id)

feature_array = np.array(feature_list)
features = torch.from_numpy(feature_array)

target_array = np.array(target_list)
targets = torch.from_numpy(target_array)

train = TensorDataset(features, targets)
trainloader = DataLoader(train, batch_size=4, shuffle=True, num_workers=2)

def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

feature_list = []
target_list = []

classes = ('Test')

image_folder = glob.iglob("../processed_images/Full_Size/Test/*.jpg")
image_folder = list(image_folder)

target_test = [2,2,1,3,3,2,2,2,2,2,2,3,2,3,2,3,1,2,2,3,1,2,2,1,2,1,2,3,2,3,1,2,2,1,2,3,2,2,2,1,3,2,2,2,1,3,1,1,3,3,3,1,3,1,3,3,2,2,2,1,3,2,3,2,2,1,3,3,3,3,1,2,2,1,3,1,3,2,3,1,2,2,2,3,1,2,3,3,2,3,1,2,3,2,2,2,1,3,2,3,1,2,2,1,2,2,2,2,3,3,2,1,3,3,2,2,2,2,3,2,2,2,3,1,3,2,2,2,2,2,2,3,3,3,2,2,2,2,2,2,2,3,1,3,2,3,2,2,2,3,1,3,2,3,1,3,2,3,3,2,1,2,2,2,2,1,1,3,1,2,2,3,2,1,2,3,2,2,3,2,3,1,3,2,3,3,2,2,2,3,2,2,3,1,2,1,1,2,3,3,2,2,2,2,2,3,2,3,3,3,3,2,2,3,2,2,2,2,2,2,2,3,3,3,3,2,2,2,2,2,2,1,1,2,3,2,2,2,3,2,3,2,2,2,3,2,2,2,1,1,2,1,3,3,3,2,3,2,3,3,1,3,2,2,1,3,3,2,1,2,3,2,3,3,2,3,2,2,3,3,2,2,2,3,3,2,2,3,3,1,2,2,2,1,3,2,2,2,3,1,2,3,2,1,1,2,3,3,1,3,3,3,1,2,2,2,1,1,2,2,2,2,2,3,2,3,3,2,3,3,1,1,2,3,3,3,1,2,2,3,2,2,1,3,1,1,2,2,2,2,2,2,2,3,2,1,1,2,2,3,2,2,2,2,3,2,3,1,2,2,2,1,3,3,2,2,3,1,2,3,3,2,1,2,2,2,3,3,2,2,2,2,2,3,2,1,2,2,2,2,3,3,1,2,2,2,1,2,2,3,3,2,3,2,3,1,1,2,3,1,2,2,3,3,2,3,3,2,3,3,1,1,3,3,2,3,2,2,2,2,3,2,2,2,2,1,2,2,2,1,3,2,3,1,2,3,2,1,1,2,1,2,2,3,2,2,2,1,2,3,2,2,1,2,2,2,2,1,2,3,3,3,1,2,3,3,2,2,2,1,2,3,2,1,2,2,3,2,2,3,2,2,3,2,3,2,2,2,2,2,2,3]

image_folder = sorted(image_folder, key = lambda x:int(x[int(len("../processed_images/Full_Size/Test/")):-8]))

for file_index, filename in enumerate(image_folder):
    piexif.remove(filename)
    image = Image.open(filename)
    try:
        image = scipy.misc.imresize(image, (256, 256))
    except ValueError:
        continue 
    image = np.array(image)
    image = np.swapaxes(image,0,2)
    feature_list.append(image)
    target_list.append(target_test[file_index])
    # print(image_folder[file_index])

feature_array = np.array(feature_list)
features = torch.from_numpy(feature_array)

for epoch in range(100):  
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        labels = labels.long()
        labels = labels - 1

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())

        # print(outputs)
        # print(labels)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    # Iterate through Type 1 image files 
    running_loss = 0.0

    # len(features)
    correct = 0
    total = 0 

    outputs = np.zeros((len(features),3))

    for i in range(0,len(features)):
        torch.manual_seed(i)
        output = net(Variable(features[i:i+1]).float().cuda())
        outputs[i] = (softmax(output.data.cpu().numpy())[0])

    running_loss = 0
    correct = 0
    total = 0

    for i in range(len(outputs)):
        if((outputs[i].argmax() + 1) != target_test[i]):
            # Add the log of the probability to the running loss
            running_loss += math.log1p(outputs[i][outputs[i].argmax()])
        else:
            correct += 1
        total += 1

    print(running_loss/len(outputs))
    print(correct/total)

    torch.save(net.state_dict(), '../classifier/Neural_Networks/NEW_Deep_CNN_ALL_' + str(epoch) + '.pth')

print('Finished Training')

stop = timeit.default_timer()

print(stop - start)