import mindspore.train.model as Model
import logging
import numpy as np
from mindspore import Tensor
import mindspore as ms
import types
import functools
from functools import wraps
import mindspore.nn as nn

model = None
orig_constr = None
flag_first = True
mod_constr = None

network_ = None
loss_fn_ = None
optimizer_ = None
metrics_ = None
eval_network_ = None
eval_indexes_ = None
amp_level_ = None
boost_level_ = None
kwargs_ = None
train_model = None

backup_Model = Model.Model.__init__
def override_Model(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None,
                 amp_level="O0", boost_level="O0", **kwargs):
    global network_
    global loss_fn_
    global optimizer_
    global metrics_
    global eval_network_
    global eval_indexes_
    global amp_level_
    global boost_level_
    global kwargs_
    global train_model

    network_ = network
    loss_fn_ = loss_fn
    optimizer_ = optimizer
    metrics_ = metrics
    eval_network_ = eval_network
    eval_indexes_ = eval_indexes
    amp_level_ = amp_level
    boost_level_ = boost_level
    kwargs_ = kwargs

    return backup_Model(self, network, loss_fn, optimizer, metrics, eval_network, eval_indexes, amp_level, boost_level, **kwargs)
Model.Model.__init__ = override_Model

class ModifiedDense(nn.Dense):
    def __init__(self, *args, **kwargs):
        super(ModifiedDense, self).__init__(*args, **kwargs)

    def construct(self, x):
        print('Hello')  # Дополнительная логика
        return super().construct(x)

temp = None

backup_check_network_mode = Model.Model._check_network_mode
def override_check_network_mode(self, network, is_train):
    global train_model
    global temp
    ret_train_network = backup_check_network_mode(self, network, is_train)
    if self._current_step_num == 10:
        # modified_dense_layer = ModifiedDense(**vars(network_.fc2))
        temp = network_.fc2
        modified_dense_layer = ModifiedDense(in_channels=network_.fc2.in_channels,
                                     out_channels=network_.fc2.out_channels,
                                     weight_init=network_.fc2.weight,
                                     bias_init=network_.fc2.bias,
                                     has_bias=network_.fc2.has_bias,
                                     activation=network_.fc2.activation)
        network_.fc2 = modified_dense_layer
        self.__init__(network_, loss_fn_, optimizer_, metrics_, eval_network_, eval_indexes_, amp_level_, boost_level_, **kwargs_)
    elif self._current_step_num == 11:
        network_.fc2 = temp
        self.__init__(network_, loss_fn_, optimizer_, metrics_, eval_network_, eval_indexes_, amp_level_, boost_level_, **kwargs_)

    return ret_train_network
Model.Model._check_network_mode = override_check_network_mode


# mindspore.nn.grad.cell_grad
# mindspore.nn.wrap.cell_wrap -> TrainOneStepCell
