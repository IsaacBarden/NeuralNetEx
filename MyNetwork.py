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
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

def load_data(filename="NeuralNetEx\\mnist_expanded.pkl.gz", dataset="training", **kwargs):
    '''
    Partially pulled from DeepNetwork.py

    Opens MNIST dataset, with an optional specific filename, returning the
    training, validation, or testing dataset depending on the value in the set parameter.

    Args:

    filename: Location of dataset file.   

    dataset: Grabs that part of the dataset. Values = "training", "validation", "testing"

    batch_size: The batch size specified in run_nn.

    NOTE: Torchvision includes the MNIST dataset, and can be more easily loaded from there.
    However, I'm loading it from an existing file for the purposes of learning. 
    '''

    f = gzip.open(filename)
    if(dataset.lower() == "training"):
        data = torch.utils.data.DataLoader(pickle.load(f, encoding="latin1")[0], **kwargs)
    elif(dataset.lower() == "validation"):
        data = torch.utils.data.DataLoader(pickle.load(f, encoding="latin1")[1], **kwargs)
    elif(dataset.lower() == "testing"):
        data = torch.utils.data.DataLoader(pickle.load(f, encoding="latin1")[2], **kwargs)
    else:
        raise ValueError("Invalid dataset name")
    f.close()
    return data

def train(model, device, optimizer, epoch, train_data, log_interval, dry_run):
    model.train()
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        batch_count = batch_idx = len(data)
        data_length = len(train_data.dataset)
        processed_percent = 100. * batch_idx / len(train_data)
        if batch_idx % log_interval == 0:
            print(f"~Training~ Epoch {epoch}: [{batch_count}/{data_length} ({processed_percent:.0f})]\tLoss: {loss.item():.4f}")
            if dry_run:
                break

def test(model, device, test_data):
    device = torch.device("cpu")
    model = Net().to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_data.dataset)
    dataset_length = len(test_data.dataset)
    correct_percentage = 100. * correct / dataset_length

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{dataset_length}, {correct_percentage:.2f}%\n")


def run_nn(batch_size=10, test_batch_size=1000, epochs=60, learning_rate=0.03, lmbda=0.1,
           use_cuda=True, dry_run=False, seed=1, log_interval=100, save_model=False, do_testing=True):
    '''
    Main function. Creates the neural network, applies the given arguments, trains, tests, saves.

    Args:
    
    batch_size: batch size for training (default = 10)

    test_batch_size: batch size for testing (default = 1000)

    epochs: number of epoch to train the nn for (default = 60)

    learning_rate: nn's learning rate (default = 1.0)

    lmbda: L2 regularization, defined as weight_decay in pytorch's SGD (default = 0.1)

    use_cuda: if True, use GPU. If False, use CPU (default = True)
    
    dry_run: Run through nn exactly once (default = False)

    seed: nn parameter randomization seed (default = 1)

    log_interval: Log progress every log_interval batches (default = 100)

    save_model: Saves nn to a file after training (default = False)

    do_testing: If True, send nn through the testing set and log performance (default = True)
    '''

    #Make sure that cuda can be used before setting to device
    #use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    #If using CUDA, optimize data loading. more info at https://pytorch.org/docs/stable/data.html
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    torch.manual_seed(seed)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    training_data = load_data(dataset="training", **train_kwargs)
    if do_testing:
        testing_data = load_data(dataset="testing", **test_kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lmbda)
    scheduler = ReduceLROnPlateau(optimizer)

    for epoch in range(1, epochs+1):
        train(model, device, optimizer, epoch, training_data, log_interval, dry_run)
        if do_testing:
            test(model, device, testing_data)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "PytorchMnistNetwork.pt")

if __name__ == '__main__':
    run_nn(epochs=5, use_cuda = False)