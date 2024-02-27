import torch
import numpy as np
import torch.nn as nn
import logging

class GradHook:
    def __init__(self, num_faults, time, boarder):
        self.num_faults = num_faults
        self.counter = 0
        self.boarder = boarder
        self.time = time
    def hook(self, module, grad_input, grad_output):
        self.counter += 1
        if self.time > 0 and self.counter >= self.boarder:
            self.time -= 1
            grad_input_copy = [g.clone() for g in grad_input]
            for k in range(self.num_faults):
                list_ind = torch.tensor([])
                for i in range(len(grad_input[0].shape)):
                    list_ind = torch.cat((list_ind, torch.randint(0, grad_input[0].shape[i], (1,))))
                grad_input_copy[0][tuple(np.int32(list_ind))] = 10
            return tuple(grad_input_copy)


class GradHookTensor:
    def __init__(self, num_faults, time, boarder):
        self.num_faults = num_faults
        self.counter = 0
        self.boarder = boarder
        self.time = time
    def hook(self, module, grad_input, grad_output):
        self.counter += 1
        if self.time > 0 and self.counter >= self.boarder:
            self.time -= 1
            for k in range(self.num_faults):
                list_ind = torch.tensor([])
                for i in range(len(grad_input[0].shape)):
                    list_ind = torch.cat((list_ind, torch.randint(0, grad_output[0].shape[i], (1,))))
                grad_output[0][tuple(np.int32(list_ind))] = 10
        

class ForwardHook:
    def __init__(self, num_faults, time, boarder):
        self.num_faults = num_faults
        self.boarder = boarder
        self.counter = 0
        self.time = time
    def hook(self, module, input, output):
        self.counter += 1
        if self.time > 0 and self.counter >= self.boarder:
            print('fault injected')
            self.time -= 1
            with torch.no_grad():
                weight_copy = module.weight.data.clone()
                for k in range(self.num_faults):
                    list_ind = torch.tensor([])
                    for i in range(len(module.weight.shape)):
                        list_ind = torch.cat((list_ind, torch.randint(0, module.weight.shape[i], (1,))))
                    weight_copy[tuple(np.int32(list_ind))] = 1
                    module.weight.data = nn.Parameter(weight_copy)


class ForwardHookTensor:
    def __init__(self, num_faults, time, boarder):
        self.num_faults = num_faults
        self.boarder = boarder
        self.counter = 0
        self.time = time
    def hook(self, module, input, output):
        self.counter += 1
        if self.time > 0 and self.counter >= self.boarder:
            print('fault injected')
            self.time -= 1
            with torch.no_grad():
                for k in range(self.num_faults):
                    list_ind = torch.tensor([])
                    for i in range(len(module.weight.shape)):
                        list_ind = torch.cat((list_ind, torch.randint(0, module.weight.shape[i], (1,))))
                    module.weight.data[tuple(np.int32(list_ind))] = 1