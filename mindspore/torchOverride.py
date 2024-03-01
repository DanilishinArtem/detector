import mindspore.nn.cell as nn
import logging
import mindspore.train.model as Model
from my_hooks import create_forward_hook
from mindspore.rewrite import SymbolTree
import mindspore.nn as nd


model_PYNATIVE_MODE = None
model_GRAPH_MODE = None
GRAPH = None
current_step = 0

from mindspore.rewrite import SymbolTree, Node, ScopedValue

_train_network = None
_jit_config_dict = None

backup_GRAPH_MODE = Model.Model.__init__
def override_GRAPH_MODE(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None,
                 amp_level="O0", boost_level="O0", **kwargs):
    global _train_network
    global _jit_config_dict
    backup_GRAPH_MODE(self, network, loss_fn, optimizer, metrics, eval_network, eval_indexes, amp_level, boost_level, **kwargs)
    _train_network = self._train_network
    _jit_config_dict = self._train_network._jit_config_dict


    stree = SymbolTree.create(self._train_network.network._backbone)
    relu_node = stree.get_node("relu")
    
    new_relu_cell = nd.ReLU()
    def new_func_construct(self, x):
        print('hello from new Relu')
        return self.relu(x)
    new_relu_cell.construct = new_func_construct
    
    new_node = Node.create_call_cell(cell=new_relu_cell, targets=[stree.unique_name("x")],
                                args=[ScopedValue.create_naming_value("x")], name="new_relu")
    stree.replace(relu_node, [new_node])
    print(self._train_network)

Model.Model.__init__ = override_GRAPH_MODE













# # backup_PYNATIVE_MODE = nn.Cell.__init__
# # def override_PYNATIVE_MODE(self, auto_prefix=True, flags=None):
# #     global model_PYNATIVE_MODE
# #     backup_PYNATIVE_MODE(self, auto_prefix=True, flags=None)
# #     if 'model' in self._cell_tag:
# #         model_PYNATIVE_MODE = self
# #         logging.warning('[Model getted]: ' + str(model_PYNATIVE_MODE._cell_tag))
# #     elif 'mindspore.nn.optim' in self._cell_tag:
# #         for name, layer in model_PYNATIVE_MODE._cells.items():
# #             if name == 'fc2':
# #                 # logging.warning('[forward hook implemented for layer ' + name + ']:')
# #                 # hook = create_forward_hook(1000, 1, 20, layer.weight, layer.bias)
# #                 # layer.register_forward_hook(hook)

# #                 logging.warning('[backward hook implemented for layer ' + name + ']:')
# #                 hook = create_forward_hook(1000, 1, 20, layer.weight, layer.bias)
# #                 layer.register_backward_hook(hook)
# # nn.Cell.__init__ = override_PYNATIVE_MODE


# backup_GRAPH_MODE = Model.Model.__init__
# def override_GRAPH_MODE(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None,
#                  amp_level="O0", boost_level="O0", **kwargs):
#     global model_GRAPH_MODE
#     backup_GRAPH_MODE(self, network, loss_fn, optimizer, metrics, eval_network, eval_indexes, amp_level, boost_level, **kwargs)
#     model_GRAPH_MODE = self
# Model.Model.__init__ = override_GRAPH_MODE


# backup__exec_preprocess = Model.Model._exec_preprocess
# def override_exec(self, is_train, dataset, dataset_sink_mode, sink_size=-1, epoch_num=1, dataset_helper=None):
#     global GRAPH
#     dataset_helper, network = backup__exec_preprocess(self, is_train, dataset, dataset_sink_mode, sink_size, epoch_num, dataset_helper)
#     logging.warning('[calculational graph overtaken]')
    
#     # place of modifying the graph (net is the graph) ----->
#     # by using methods:
#         # from mindspore.rewrite import SymbolTree
#         # stree = SymbolTree.create(net)
#         # stree.print_node_tabulate()
#         # basic methods: (https://www.mindspore.cn/docs/en/r2.2/api_python/samples/rewrite/rewrite_tutorial.html)
#             # stree.insert
#             # stree.erase
#             # stree.replace
#             # new_net = stree.get_network()
#     name, GRAPH = next(iter(network._cells.items()))

#     # stree = SymbolTree.create(GRAPH._backbone)
#     # relu_node = stree.get_node("relu")
    
#     # new_relu_cell = nd.ReLU()
#     # def new_func_construct(x):
#     #     print('hello from new Relu')
#     #     return new_relu_cell.construct(x)
#     # new_relu_cell.construct = new_func_construct
    
#     # new_node = Node.create_call_cell(cell=new_relu_cell, targets=[stree.unique_name("x")],
#     #                             args=[ScopedValue.create_naming_value("x")], name="new_relu")
#     # stree.replace(relu_node, [new_node])
#     # # GRAPH._backbone = stree.get_network()

#     return dataset_helper, network
# Model.Model._exec_preprocess = override_exec


# backup_check_network_mode = Model.Model._check_network_mode
# def override_check_network_mode(self, network, is_train):
#     global current_step
#     current_step = model_GRAPH_MODE._current_step_num
#     ret_train_network = backup_check_network_mode(self, network, is_train)
#     print('=========================================================================')
#     logging.warning('[current_step_num]: ' + str(model_GRAPH_MODE._current_step_num))

#     if model_GRAPH_MODE._current_step_num == 10:
#         stree = SymbolTree.create(ret_train_network.network._backbone)
#         relu_node = stree.get_node("relu")
        
#         new_relu_cell = nd.ReLU()
#         def new_func_construct(self, x):
#             # logging.warning('hello from new Relu')
#             # x = x * 1000
#             # return self.relu(x)
#             return None
#         new_relu_cell.construct = new_func_construct
        
#         new_node = Node.create_call_cell(cell=new_relu_cell, targets=[stree.unique_name("x")],
#                                  args=[ScopedValue.create_naming_value("x")], name="new_relu")
#         stree.replace(relu_node, [new_node])
#         ret_train_network.network._backbone = stree.get_network()
#     # print(ret_train_network)
#     return ret_train_network
# Model.Model._check_network_mode = override_check_network_mode