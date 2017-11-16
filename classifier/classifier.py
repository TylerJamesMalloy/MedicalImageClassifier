import torch
from torch.utils.serialization import load_lua

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

##____________ CLASSIFICATION __________________

net = load_lua('a.t7')

dataiter = iter(trainloader)
images, labels = dataiter.next()

images = images.float()
labels = labels.long()

outputs = net(Variable(images))

_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(10)))
