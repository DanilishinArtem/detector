import torch
import numpy as np
import torch.nn as nn
import logging

# class GradHook:
#     def __init__(self, num_faults, time, boarder):
#         self.num_faults = num_faults
#         self.counter = 0
#         self.boarder = boarder
#         self.time = time
#     def hook(self, module, grad_input, grad_output):
#         self.counter += 1
#         if self.time > 0 and self.counter >= self.boarder:
#             self.time -= 1
#             grad_input_copy = [g.clone() for g in grad_input]
#             for k in range(self.num_faults):
#                 list_ind = torch.tensor([])
#                 for i in range(len(grad_input[0].shape)):
#                     list_ind = torch.cat((list_ind, torch.randint(0, grad_input[0].shape[i], (1,))))
#                 grad_input_copy[0][tuple(np.int32(list_ind))] = 10
#             return tuple(grad_input_copy)


class GradHook_output_prehook:
    def __init__(self, num_faults, time, boarder):
        self.num_faults = num_faults
        self.counter = 0
        self.boarder = boarder
        self.time = time
    def hook(self, module, grad_output):
        self.counter += 1
        if self.time > 0 and self.counter >= self.boarder:
            # print(self.hook.my_weights[20][30])
            self.time -= 1
            grad_output_copy = [g.clone() for g in grad_output]
            for k in range(self.num_faults):
                list_ind = torch.tensor([])
                for i in range(len(grad_output[0].shape)):
                    list_ind = torch.cat((list_ind, torch.randint(0, grad_output[0].shape[i], (1,))))
                grad_output_copy[0][tuple(np.int32(list_ind))] = 1
                # grad_output[0][tuple(np.int32(list_ind))] = 1
            return tuple(grad_output_copy)


class GradHook_input_hook:
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
                grad_output[0][tuple(np.int32(list_ind))] = 999999999999999
        

# ------------------------------------------- weights hook ------------------------------------------------


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




def create_forward_hook(num_faults, time, boarder, my_weights):
    counter = 0
    my_weights.to('cuda')
    def f_ForwardHookTensor(module, input, output):
        nonlocal counter
        nonlocal num_faults
        nonlocal time
        nonlocal boarder
        nonlocal my_weights
        counter += 1
        if time > 0 and counter >= boarder:
            time -= 1
            for k in range(num_faults):
                list_ind = torch.tensor([])
                for i in range(len(my_weights.shape)):
                    list_ind = torch.cat((list_ind, torch.randint(0, my_weights.shape[i], (1,))))
                # print('[f_ForwardHookTensor.my_weights[20][20]]: ' + str(my_weights[20][20]))
                # print('[module.weight.data[20][20]]: ' + str(module.weight.data[20][20]))
                my_weights[tuple(np.int32(list_ind))] = 10
                # module.weight.data[tuple(np.int32(list_ind))] = 10000
            print('fault injected')
    return f_ForwardHookTensor