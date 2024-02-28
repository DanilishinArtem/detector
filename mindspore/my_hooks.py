import torch
import numpy as np
import torch.nn as nn
import logging
from mindspore import Tensor
        

def create_forward_hook(num_faults, time, boarder, weights, bias):
    counter = 0
    def f_ForwardHookTensor(module, input, output):
        nonlocal counter
        nonlocal num_faults
        nonlocal time
        nonlocal boarder
        nonlocal weights
        nonlocal bias
        counter += 1
        if time > 0 and counter >= boarder:
            time -= 1
            # copy_weights = weights.copy().data.asnumpy()
            # for k in range(num_faults):
            #     list_ind = torch.tensor([])
            #     for i in range(len(copy_weights.shape)):
            #         list_ind = torch.cat((list_ind, torch.randint(0, copy_weights.shape[i], (1,))))
            #     copy_weights[tuple(np.int32(list_ind))] = 1000
            # weights.set_data(Tensor(copy_weights))
            # logging.warning('fault injected')

            copy_bias = bias.copy().data.asnumpy()
            for k in range(num_faults):
                list_ind = torch.tensor([])
                for i in range(len(copy_bias.shape)):
                    list_ind = torch.cat((list_ind, torch.randint(0, copy_bias.shape[i], (1,))))
                copy_bias[tuple(np.int32(list_ind))] = 10
            bias.set_data(Tensor(copy_bias))
            logging.warning('fault injected')

    return f_ForwardHookTensor