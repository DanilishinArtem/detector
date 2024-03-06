import mindspore.train.model as Model
import logging
import numpy as np
from mindspore import Tensor
import mindspore as ms
import types
import functools

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




backup_check_network_mode = Model.Model._check_network_mode
def override_check_network_mode(self, network, is_train):
    global train_model

    # def create_mod_constr(orig_constr):
    #     def foo(self, x):
    #         nonlocal orig_constr
    #         return orig_constr(x)
    #     return foo
    
    # def modif_constr(self, x):
    #     logging.warning('hello from forward')
    #     return orig_constr(self, x)

    ret_train_network = backup_check_network_mode(self, network, is_train)
    logging.warning('========================================')
    logging.warning('current step: ' + str(self._current_step_num))
    # if self.current_step_num == 10:
    #     logging.warning('[weight fault injected to the layer fc2]')
    #     copy_weight = self.network.fc2.weight.copy().data.asnumpy()
    #     self._network.fc2.weight.set_data(Tensor(np.full_like(copy_weight, 1000)))

    if self._current_step_num == 10:
        temp = network_.fc2.construct
        @functools.wraps(temp)
        def construct(x):
            # print('hello')
            return temp(x * 0)
        funcType = types.MethodType
        network_.fc2.construct = funcType(ms.jit(construct), network_.fc2)
        self.__init__(network_, loss_fn_, optimizer_, metrics_, eval_network_, eval_indexes_, amp_level_, boost_level_, **kwargs_)
        


    return ret_train_network
Model.Model._check_network_mode = override_check_network_mode