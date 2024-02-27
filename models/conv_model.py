import torch.nn as nn
import torch.nn.functional as F
# from hooks.hooks import GradHook
from hooks.hooks import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc1.register_backward_hook(GradHook(1000, 1, 2000).hook)
        # self.fc1.register_backward_hook(GradHookTensor(1000, 1, 2000).hook)
        # self.fc1.register_forward_hook(ForwardHook(1000, 1, 2000).hook)
        # self.fc1.register_forward_hook(ForwardHookTensor(1000, 1, 2000).hook)



        self.fc1.register_full_backward_pre_hook(GradHook_output_prehook(1000, 1, 2000).hook)
        # self.fc1.register_full_backward_hook(GradHook_input_hook(1000, 1, 2000).hook)

        # register_full_backward_hook register_backward_hook 
        # self.fc1.register_backward_hook(GradHookTensor(1000, 100, 2000).hook)

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
    

# GradHook_output_prehook -> register_full_backward_pre_hook
# GradHook_input_hook -> register_full_backward_hook