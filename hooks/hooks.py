import torch
import numpy as np
import logging

class GradHook:
    def __init__(self, num_faults, boarder):
        self.num_faults = num_faults
        self.counter = 0
        self.boarder = boarder

    def hook(self, module, grad_input, grad_output):
        self.counter += 1
        if self.num_faults > 0 and self.counter >= self.boarder:
            self.num_faults -= 1
            grad_input_copy = [g.clone() for g in grad_input]
            for k in range(self.num_faults):
                list_ind = torch.tensor([])
                for i in range(len(grad_input[0].shape)):
                    list_ind = torch.cat((list_ind, torch.randint(0, grad_input[0].shape[i], (1,))))
                grad_input_copy[0][tuple(np.int32(list_ind))] = 1
                # print("Modified gradients:", grad_input_copy[0][tuple(np.int32(list_ind))])
            return tuple(grad_input_copy)