#Creating and training the same kind of NN as in the final network 
#from chapter 6, this time using pytorch to create the network.

#Resources used: 
# http://neuralnetworksanddeeplearning.com/chap6.html
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
# https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
# https://github.com/pytorch/examples/blob/master/mnist/main.py

import pickle
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #1 input channel, 20 output channels, 5x5 convolution from 28x28 images to 24x24 images
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)

        #20 input channels, 40 output channels, 5x5 convolution from 12x12 to 8x8 images
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)

        #first fully connected layer, 40 4x4 images input, 1000 output features
        self.fc1 = nn.Linear(in_features=40*4*4, out_features=1000)
        
        #second fully connected layer, 1000 input featues, 1000 output features
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)

        #output layer, 1000 input features, 10 output possibilities (one for each number)
        self.out = nn.Linear(in_features=1000, out_features=10)

        #define 50% dropout for use with all linear layers
        self.dropout = nn.Dropout(p=0.5)

    #x represents the data
    def forward(self, x):
        #2x2 Max pooling of 24x24 image into 12x12 image, conv layer will use ReLU
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) 

        #2x2 max pooling of 8x8 image into 4x4 image, use ReLU
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        #flatten convolutional layers into a linear format
        x = torch.flatten(x, start_dim=1)

        #pass data through 50% dropout then first fully connected layer
        x = F.relu(self.fc1(self.dropout(x)))

        #pass data through 50% dropout then second fully connected layer
        x = F.relu(self.fc2(self.dropout(x)))

        #pass data through 50% dropout then get softmax of output
        output = F.softmax(self.out(self.dropout(x)), dim=1)

        return output

def load_data(filename="NeuralNetEx\\mnist_expanded.pkl.gz", dataset="training", batch_size=1):
    '''
    Partially pulled from DeepNetwork.py

    Opens MNIST dataset, with an optional specific filename, returning the
    training, validation, or testing dataset depending on the value in the set parameter.

    dataset = "training", "validation", "testing"

    NOTE: Torchvision includes the MNIST dataset, and can be more easily loaded from there.
    However, I'm loading it from an existing file for the purposes of learning. 
    '''

    f = gzip.open(filename)
    if(dataset.lower() == "training"):
        data = torch.utils.data.DataLoader(pickle.load(f, encoding="latin1")[0], batch_size=batch_size)
    elif(dataset.lower() == "validation"):
        data = torch.utils.data.DataLoader(pickle.load(f, encoding="latin1")[1], batch_size=batch_size)
    elif(dataset.lower() == "testing"):
        data = torch.utils.data.DataLoader(pickle.load(f, encoding="latin1")[2], batch_size=batch_size)
    else:
        raise ValueError("Invalid dataset name")
    f.close()
    return data

load_data()