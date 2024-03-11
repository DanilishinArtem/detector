import mindspore.train.model as Model
import logging
import numpy as np
from mindspore import Tensor
import mindspore as ms
import types
import functools
from functools import wraps
import mindspore.nn as nn
import mindspore.common.dtype as mstype

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

# def insert_dense_layers(network, dense_layers):
    
#     def explore_cells(cell):
        # if isinstance(cell, nn.Dense):
#             # logging.warning('changing layer: ' + str(cell) + ' to: ' + str(ModifiedDense))
#             dense_layers.append(cell)
#             modified_dense_layer = ModifiedDense(
#                 in_channels=cell.in_channels,
#                 out_channels=cell.out_channels,
#                 weight_init=cell.weight,
#                 bias_init=cell.bias,
#                 has_bias=cell.has_bias,
#                 activation=cell.activation
#             )
#             return modified_dense_layer  # Возвращаем модифицированный слой
#         elif hasattr(cell, 'cells'):
#             # Создаем новый список для обновленных подячеек
#             new_cells = []
#             for subcell in cell.cells():
#                 # Рекурсивно обходим подмодели и заменяем слои в них
#                 new_subcell = explore_cells(subcell)
#                 new_cells.append(new_subcell)
#             # Обновляем подячейки в текущей ячейке
#             cell.cells = lambda: new_cells
#         return cell

#     # Начинаем обход с корневой ячейки
#     network = explore_cells(network)
#     return dense_layers

# def restore_dense_layers(network, dense_layers):
#     counter = 0
#     def explore_cells(cell):
#         nonlocal counter
#         if isinstance(cell, nn.Dense):
#             cell = dense_layers[counter]
#             counter += 1
#         if hasattr(cell, '_cells'):
#             for subcell in cell._cells():
#                 explore_cells(subcell)
#     explore_cells(network)


backup_check_network_mode = Model.Model._check_network_mode
def override_check_network_mode(self, network, is_train):
    global train_model
    global temp
    ret_train_network = backup_check_network_mode(self, network, is_train)
    if self._current_step_num == 1:
    # if self._current_step_num == 0 or self._current_step_num == 1 or self._current_step_num == 2:
        class ModifiedDense(nn.Dense):
            def __init__(self, *args, **kwargs):
                super(ModifiedDense, self).__init__(*args, **kwargs)

            def construct(self, x):
                print('Hello')  # Дополнительная логика
                return super().construct(x)
        logging.warning('[injecting dense layer...]')
        # temp = insert_dense_layers(network_, temp)
        # modified_dense_layer = ModifiedDense(**vars(network_.fc2))
        temp = network_.poetry.bert.bert_encoder.layers[0].attention.output.dense
        # temp = network_.poetry.bert.bert_encoder.layers[11].intermediate
        modified_dense_layer = ModifiedDense(in_channels=temp.in_channels,
                                     out_channels=temp.out_channels,
                                     weight_init=temp.weight,
                                     bias_init=temp.bias,
                                     has_bias=temp.has_bias,
                                     activation=temp.activation,
                                     dtype=mstype.float32)
        network_.poetry.bert.bert_encoder.layers[0].attention.output.dense = modified_dense_layer
        # network_.poetry.bert.bert_encoder.layers[11].intermediate = modified_dense_layer
        self.__init__(network_, loss_fn_, optimizer_, metrics_, eval_network_, eval_indexes_, amp_level_, boost_level_, **kwargs_)
        # self.__init__(network_)

    # elif self._current_step_num == 4:
    #     restore_dense_layers(network_, temp)
    #     self.__init__(network_, loss_fn_, optimizer_, metrics_, eval_network_, eval_indexes_, amp_level_, boost_level_, **kwargs_)

    return ret_train_network
Model.Model._check_network_mode = override_check_network_mode


# mindspore.nn.grad.cell_grad
# mindspore.nn.wrap.cell_wrap -> TrainOneStepCell

