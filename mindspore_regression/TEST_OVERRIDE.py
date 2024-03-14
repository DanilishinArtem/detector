import mindspore.train.model as Model
import logging
import mindspore.nn as nn
import time
from mindspore import jit

network_ = None

backup_Model = Model.Model.__init__
def override_Model(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None, amp_level="O0", boost_level="O0", **kwargs):
    global network_
    network_ = network
    return backup_Model(self, network, loss_fn, optimizer, metrics, eval_network, eval_indexes, amp_level, boost_level, **kwargs)
Model.Model.__init__ = override_Model

# ----------------------------------------------
faulting_layer = None
start_of_faulting = 10 # np.random.randint(100)
time_of_faulting = 1

# matmul fault ----------------------->
def mod_matmul(x, weight):
    x[1,:] = 1
    result = faulting_layer.matmul(x, weight)
    # result[0:25,:] = 1
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
        self.weight[0:25,:] = 1
        return super().construct(x)

# data fault ----------------------->
class ModifiedDenseDataFault(nn.Dense):
    def __init__(self, *args, **kwargs):
        super(ModifiedDenseDataFault, self).__init__(*args, **kwargs)

    def construct(self, x):
        x[0:25,:] = 1
        return super().construct(x)
    

from mindspore import context
# dot -Tpdf ./graph/ModelDigraph_0893.dot -o ./graph.pdf

backup_check_network_mode = Model.Model._check_network_mode
def override_check_network_mode(self, network, is_train):
    global start_of_faulting
    global time_of_faulting
    global faulting_layer
    global network_
    # time.sleep(0.5)

    if self._current_step_num == start_of_faulting:
        context.set_context(save_graphs=3, save_graphs_path="./graph")
        faulting_layer = self._train_network.network._backbone.fc
        logging.warning('[injecting dense layer...]')

        attributes_instance1 = vars(faulting_layer)
        
        modified_dense_layer = ModifiedDenseMatmulFault(in_channels=faulting_layer.in_channels, out_channels=faulting_layer.out_channels, weight_init=faulting_layer.weight, bias_init=faulting_layer.bias, has_bias=faulting_layer.has_bias, activation=faulting_layer.activation)
        # modified_dense_layer = ModifiedDenseWeightFault(in_channels=faulting_layer.in_channels, out_channels=faulting_layer.out_channels, weight_init=faulting_layer.weight, bias_init=faulting_layer.bias, has_bias=faulting_layer.has_bias, activation=faulting_layer.activation)
        # modified_dense_layer = ModifiedDenseDataFault(in_channels=faulting_layer.in_channels, out_channels=faulting_layer.out_channels, weight_init=faulting_layer.weight, bias_init=faulting_layer.bias, has_bias=faulting_layer.has_bias, activation=faulting_layer.activation)
        
        
        for attr_name, attr_value in attributes_instance1.items():
            if attr_name != 'matmul':
                setattr(modified_dense_layer, attr_name, attr_value)        
        self._train_network._create_time = int(time.time() * 1e9)
        self._train_network.network._backbone._cells['fc'] = modified_dense_layer



    if self._current_step_num == start_of_faulting + time_of_faulting:
        context.set_context(save_graphs=3, save_graphs_path="./graph_res")
        self._train_network.network._backbone._cells['fc'] = faulting_layer
        self._train_network._create_time = int(time.time() * 1e9)

    ret_train_network = backup_check_network_mode(self, network, is_train)
    return ret_train_network

Model.Model._check_network_mode = override_check_network_mode


