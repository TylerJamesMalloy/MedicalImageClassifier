import torch
import torch.nn as nn

classes = ('Type_1','Type_2','Type_3')

##____________ CLASSIFICATION __________________

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
net.load_state_dict(torch.load('TrainedNN_1000.pt'))

images = []
target_list = []

# Still need to do post-pre-processing-processing
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
        images.append(image)
        target_list.append(i)

outputs = net(Variable(images))

_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(10)))
