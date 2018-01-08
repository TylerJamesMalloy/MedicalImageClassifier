# Medical Image Classifier

## Team
Tyler Malloy, Austin Wallace, Varoon Mathur, Matthias Lee

## Project 
This project was created to compare the ability of a deep convolutional neural network with the newly developed [capsule network](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf) outlined in NIPS 2017. The domain of this comparison is image classification of cervical cancer with image data retrieved from the [Intel & MobileODT Cervical Cancer Screening](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening) Kaggle competition.

## Motivation
Capsule networks, as described by Hinton et. al (2017), was recently introduced as a network that is more robust to detecting how images and objects are positioned and more easily able to generalize these spatial relationships. By nesting another layer of neurons within a layer in an NN (a capsule), lower and higher-level capsules can connect to provide better feature detection in lower layers of the image, and is a much more effective method than the max-pooling mechanism in CNNs (which essentially reduces the spatial size of the image in order to pick the largest feature, considered to be crude). While this new methodology has shown to outperform CNNs on the classical MNIST data set of images, we wished to understand their comparison on a most robust image dataset.

### Data Pre-Processing 

The pre-processing step was integral to the performance of our networks. 

### Deep CNN Structure

The structure of our Deep CNN was modified from the [py-torch source turotials](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py) on CNNs. 
This structure was edited to fit the dimensions required for our tests on images normalized to 256x256 after pre-processing, with 3 possible classifications.  

![Deep CNN Structure](https://i.imgur.com/pMZmyXL.png)

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

### Capsule Network Structure

The struecute of our Capsule Network was modified from a test of the [capsule network on the MNIST dataset](https://github.com/gram-ai/capsule-networks). 
This structure was edited to fit the dimensions required for our tests on images reduced to 32x32 after pre-processing, with 3 possible classifications.  

![Capsule Network Structure](https://i.imgur.com/7qcCQQI.png) 

    class CapsuleLayer(nn.Module):
        def __init__(self, num_capsules, num_routes, in_channels, out_channels,
                 kernel_size=None, stride=None, num_iterations=3):
        super().__init__()
    
        self.num_routes = num_routes
        self.num_iterations = num_iterations
    
        self.num_capsules = num_capsules
    
        if num_routes != -1:
            self.route_weights = nn.Parameter(
                torch.randn(num_capsules, num_routes,
                            in_channels, out_channels)
            )
    
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=0)
                 for _ in range(num_capsules)
                 ]
            )
    
    def forward(self, x):
        # If routing is defined
        if self.num_routes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
    
            logits = Variable(torch.zeros(priors.size()))
    
            # Routing algorithm
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = squash_vector(
                    probs * priors).sum(dim=2, keepdim=True)
    
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
    
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1)
                       for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = squash_vector(outputs)
    
        return outputs
    
    
    class CapsuleNet(nn.Module):
      def __init__(self):
          super().__init__()
    
          self.conv1 = nn.Conv2d(
              in_channels=1, out_channels=256, kernel_size=9, stride=1)
          self.primary_capsules = CapsuleLayer(
              8, -1, 256, 32, kernel_size=9, stride=2)
    
          # 3 is the number of classes
          self.digit_capsules = CapsuleLayer(3, 32 * 6 * 6, 8, 16)
    
          self.decoder = nn.Sequential(
              nn.Linear(16 * 3, 512),
              nn.ReLU(inplace=True),
              nn.Linear(512, 1024),
              nn.ReLU(inplace=True),
              nn.Linear(1024, 784),
              nn.Sigmoid()
          )
    
      def forward(self, x, y=None):
          x = F.relu(self.conv1(x), inplace=True)
          x = self.primary_capsules(x)
          x = self.digit_capsules(x).squeeze().transpose(0, 1)
    
          classes = (x ** 2).sum(dim=-1) ** 0.5
          classes = F.softmax(classes)
    
          if y is None:
              # In all batches, get the most active capsule
              _, max_length_indices = classes.max(dim=1)
              y = Variable(torch.eye(3)).index_select(
                  dim=0, index=max_length_indices.data)
    
          reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
          return classes, reconstructions


### Data Selection and Network Comparison

Due to the structure of the data set available from the source Kaggle competition it was important to test our networks on different levels of data sets and ensure the best perforamnce. 
The data sets from Kaggle included an initial set of images and an additional set that contained some imges from the same cancer screening and some images that were overlaid in a green filter. 
The possible levels of the data set that we investigated were all the available images, only the original image set, and a reduced set of all the images that did not include green images. 
The results from these comapisons are shown in the following graph. 

![All Results](https://i.imgur.com/zMyAY8M.png)

## Results 

![Capsule Network Results](https://i.imgur.com/rjEJibM.png)
