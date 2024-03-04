import torch
import numpy as np
import torch.nn as nn
import logging
from mindspore import Tensor
from mindspore.ops import operations as P

def create_forward_hook(num_faults, time, boarder, weights, bias):
    counter = 0
    def f_ForwardHookTensor(cell_id, input, output):
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


def create_matmul_hook(layer, time, boarder):
    counter = 0
    stock_matmul = layer.matmul
    def new_matmul(x1, x2):
        print('modified matmul')
        print(x1.shape, x2.shape)
        print('x1:\n')
        print(x1)
        print('x2:\n')
        print(x2)
        return stock_matmul(x1, x2)
    def f_MatMulHook(cell_id, input):
        nonlocal counter, time, boarder, stock_matmul, new_matmul
        counter += 1
        if time > 0 and counter == boarder:
            time -= 1
            layer.matmul = new_matmul
        else:
            layer.matmul = stock_matmul
    return f_MatMulHook

