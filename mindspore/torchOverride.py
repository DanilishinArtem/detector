import mindspore.train.model as Model
import logging
import numpy as np
from mindspore import Tensor
import mindspore as ms
import types
import functools
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import context


network_ = None

backup_Model = Model.Model.__init__
def override_Model(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None, amp_level="O0", boost_level="O0", **kwargs):
    global network_
    network_ = network
    return backup_Model(self, network, loss_fn, optimizer, metrics, eval_network, eval_indexes, amp_level, boost_level, **kwargs)
Model.Model.__init__ = override_Model

# ----------------------------------------------

faulting_layer = None
start_of_faulting = 5 # np.random.randint(100)
time_of_faulting = 2


# matmul fault ----------------------->
def mod_matmul(x, weight):
    print('injecting matmul fault...')
    result = faulting_layer.matmul(x, weight)
    print('result of matmul before injection')
    print(result)
    print('result of matmul after injection')
    result[0:25,:] = 1000
    print(result)
    return result

class ModifiedDenseMatmulFault(nn.Dense):
    def __init__(self, *args, **kwargs):
        super(ModifiedDenseMatmulFault, self).__init__(*args, **kwargs)
        self.matmul = mod_matmul

    def construct(self, x):
        return super().construct(x)

# weight fault ----------------------->
class ModifiedDenseWeightFault(nn.Dense):
    def __init__(self, *args, **kwargs):
        super(ModifiedDenseWeightFault, self).__init__(*args, **kwargs)

    def construct(self, x):
        print('injecting weight fault...')
        print('weights before injection')
        print(self.weight)
        print('weight after injection')
        self.weight[0:25,:] = 1000
        print(self.weight)
        return super().construct(x)

# data fault ----------------------->
class ModifiedDenseDataFault(nn.Dense):
    def __init__(self, *args, **kwargs):
        super(ModifiedDenseDataFault, self).__init__(*args, **kwargs)

    def construct(self, x):
        print('injecting data fault...')
        print('datas before injection')
        print(x)
        print('data after injection')
        x[0:25,:] = 1000
        print(x)
        return super().construct(x)
    


backup_check_network_mode = Model.Model._check_network_mode
def override_check_network_mode(self, network, is_train):
    global start_of_faulting
    global time_of_faulting
    global faulting_layer
    global network_
    
    ret_train_network = backup_check_network_mode(self, network, is_train)
    if self._current_step_num == start_of_faulting:
        faulting_layer = self._train_network.network._backbone.fc2
        logging.warning('[injecting dense layer...]')

        modified_dense_layer = ModifiedDenseMatmulFault(in_channels=faulting_layer.in_channels,out_channels=faulting_layer.out_channels,weight_init=faulting_layer.weight,bias_init=faulting_layer.bias,has_bias=faulting_layer.has_bias,activation=faulting_layer.activation)
        # modified_dense_layer = ModifiedDenseWeightFault(in_channels=faulting_layer.in_channels,out_channels=faulting_layer.out_channels,weight_init=faulting_layer.weight,bias_init=faulting_layer.bias,has_bias=faulting_layer.has_bias,activation=faulting_layer.activation)
        # modified_dense_layer = ModifiedDenseDataFault(in_channels=faulting_layer.in_channels,out_channels=faulting_layer.out_channels,weight_init=faulting_layer.weight,bias_init=faulting_layer.bias,has_bias=faulting_layer.has_bias,activation=faulting_layer.activation)

        self._train_network.network._backbone.fc2 = modified_dense_layer
        self._train_network = self._build_train_network()
    elif self._current_step_num == start_of_faulting + time_of_faulting:
        self._train_network.network._backbone.fc2 = faulting_layer
        # self._train_network = self._build_train_network()
    
    return ret_train_network
Model.Model._check_network_mode = override_check_network_mode