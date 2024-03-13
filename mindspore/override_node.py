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
from mindspore.rewrite import SymbolTree, ScopedValue, Node, NodeType, Replacement, PatternEngine, PatternNode, TreeNodeHelper

network_ = None

backup_Model = Model.Model.__init__
def override_Model(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None, amp_level="O0", boost_level="O0", **kwargs):
    global network_
    network_ = network
    return backup_Model(self, network, loss_fn, optimizer, metrics, eval_network, eval_indexes, amp_level, boost_level, **kwargs)
Model.Model.__init__ = override_Model

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
    
start_of_faulting = 3
time_of_faulting = 1

backup_check_network_mode = Model.Model._check_network_mode
def override_check_network_mode(self, network, is_train):
    global start_of_faulting
    global time_of_faulting
    global faulting_layer
    global network_
    from mindspore.common.api import jit

    ret_train_network = backup_check_network_mode(self, network, is_train)
    if self._current_step_num == start_of_faulting:
        faulting_layer = self._train_network.network._backbone.fc2
        logging.warning('[injecting dense layer...]')
        modified_dense_layer = ModifiedDenseDataFault(in_channels=faulting_layer.in_channels,out_channels=faulting_layer.out_channels,weight_init=faulting_layer.weight,bias_init=faulting_layer.bias,has_bias=faulting_layer.has_bias,activation=faulting_layer.activation)
        self._train_network.network._backbone.fc2._run_construct = modified_dense_layer.construct
        self._train_network.network._backbone.fc2.construct = modified_dense_layer.construct
        self._train_network.network._backbone.fc2.compile(Tensor(np.zeros((32, faulting_layer.in_channels)), mstype.float32))
        self._train_network.network._backbone.fc2.recompute()
        # self._train_network.network._backbone.fc2 = modified_dense_layer
        self._train_network = self._build_train_network()
    return ret_train_network
Model.Model._check_network_mode = override_check_network_mode