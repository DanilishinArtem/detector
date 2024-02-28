import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from learning.learner import Learner
from models.conv_model import Net
from hooks.hooks import *

def scripts_model_conv():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()

    return [trainloader, testloader, criterion]

if __name__ == '__main__':
    trainloader, testloader, criterion = scripts_model_conv()

    model = Net()
    model.to('cuda')
    hook = create_forward_hook(1000, 1, 2000, model.fc1.weight.data)
    model.fc1.register_forward_hook(hook)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    learner = Learner(model, trainloader, testloader, 1, criterion, optimizer, 'cuda')
    # learner = Learner(model, trainloader, testloader, 1, criterion, optimizer, 'cpu')

    learner.run_learning()
    # learner.run_testing()

