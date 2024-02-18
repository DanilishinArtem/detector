import torch
import numpy as np
import logging

class GradHook:
    def __init__(self, num_faults):
        self.num_faults = num_faults
    def hook(self, module, grad_input, grad_output):
        if self.num_faults > 0:
            for k in range(self.num_faults):
                list_ind = torch.tensor([])
                for i in range(len(grad_input[2].shape)):
                    list_ind = torch.cat((list_ind, torch.randint(0, grad_input[2].shape[i], (1,))))
                # grad_input[2][tuple(np.int32(list_ind))] = 1
                # grad_input[2][tuple(np.int32(list_ind))] = 0
                # grad_input[2][tuple(np.int32(list_ind))] = torch.rand([1])
            # logging.warning("[FAULT INJECTED!]")
