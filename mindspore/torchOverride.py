import mindspore.nn.cell as nn
import logging
import mindspore.train.model as Model
from my_hooks import create_forward_hook


model_PYNATIVE_MODE = None
model_GRAPH_MODE = None
GRAPH = None


backup_PYNATIVE_MODE = nn.Cell.__init__
def override_PYNATIVE_MODE(self, auto_prefix=True, flags=None):
    global model_PYNATIVE_MODE
    backup_PYNATIVE_MODE(self, auto_prefix=True, flags=None)
    if 'model' in self._cell_tag:
        model_PYNATIVE_MODE = self
        logging.warning('[Model getted]: ' + str(model_PYNATIVE_MODE._cell_tag))
    elif 'mindspore.nn.optim' in self._cell_tag:
        for name, layer in model_PYNATIVE_MODE._cells.items():
            if name == 'fc2':
                # logging.warning('[forward hook implemented for layer ' + name + ']:')
                # hook = create_forward_hook(1000, 1, 20, layer.weight, layer.bias)
                # layer.register_forward_hook(hook)

                logging.warning('[backward hook implemented for layer ' + name + ']:')
                hook = create_forward_hook(1000, 1, 20, layer.weight, layer.bias)
                layer.register_backward_hook(hook)
nn.Cell.__init__ = override_PYNATIVE_MODE


backup_GRAPH_MODE = Model.Model.__init__
def override_GRAPH_MODE(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None,
                 amp_level="O0", boost_level="O0", **kwargs):
    global model_GRAPH_MODE
    backup_GRAPH_MODE(self, network, loss_fn, optimizer, metrics, eval_network, eval_indexes, amp_level, boost_level, **kwargs)
    model_GRAPH_MODE = self
Model.Model.__init__ = override_GRAPH_MODE


backup__exec_preprocess = Model.Model._exec_preprocess
def override_exec(self, is_train, dataset, dataset_sink_mode, sink_size=-1, epoch_num=1, dataset_helper=None):
    global GRAPH
    dataset_helper, network = backup__exec_preprocess(self, is_train, dataset, dataset_sink_mode, sink_size, epoch_num, dataset_helper)
    logging.warning('[calculational graph overtaken]')
    
    # place of modifying the graph (net is the graph) ----->
    # by using methods:
        # from mindspore.rewrite import SymbolTree
        # stree = SymbolTree.create(net)
        # stree.print_node_tabulate()
        # basic methods: (https://www.mindspore.cn/docs/en/r2.2/api_python/samples/rewrite/rewrite_tutorial.html)
            # stree.insert
            # stree.erase
            # stree.replace
            # new_net = stree.get_network()
    name, GRAPH = next(iter(network._cells.items()))
    return dataset_helper, network
Model.Model._exec_preprocess = override_exec


backup_check_network_mode = Model.Model._check_network_mode
def override_check_network_mode(self, network, is_train):
    ret_train_network = backup_check_network_mode(self, network, is_train)
    print('=========================================================================')
    logging.warning('[current_step_num]: ' + str(model_GRAPH_MODE._current_step_num))
    print(str(GRAPH))
    return ret_train_network
Model.Model._check_network_mode = override_check_network_mode