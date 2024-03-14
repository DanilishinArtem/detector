import numpy as np
import mindspore
from mindspore import dataset as ds, ops
from mindspore.common.initializer import Normal
from mindspore import nn, Model, context, ParallelMode
from mindspore.train.callback import LossMonitor
import mindspore as ms
import os
from mindspore.communication.management import init, get_rank, get_group_size

import logging
from mindspore import jit

device_id = 0
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
# context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
context.set_context(device_id=device_id)
# context.set_context(save_graphs=3, save_graphs_path="./graph")

def get_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = w * x + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data

# ------------------------------------------------------------
@jit
def foo(x):
    print('hello')
    return x**2


class LinearModel(nn.Cell):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Dense(1, 1, weight_init=Normal(0.02), bias_init=Normal(0.02))
        # self.foo = foo
        # self.foo = ops.PyFunc(foo, in_types=[mindspore.float32], in_shapes=[(1,)], out_types=[mindspore.float32], out_shapes=[(1,)], name="foo", stateful=True)

    def construct(self, x):
        x = self.fc(x)
        # x = self.foo(x)
        return x

# ------------------------------------------------------------

if __name__ == '__main__':
    data_number = 10000
    batch_number = 32
    repeat_number = 1000
    lr = 0.005
    momentum = 0.9
    net = LinearModel()
    loss_cb = LossMonitor()
    ds_train = create_dataset(data_number, batch_size=batch_number, repeat_size=repeat_number)
    net_loss = nn.loss.MSELoss()
    opt = nn.Momentum(net.trainable_params(), lr, momentum)
    model = Model(net, loss_fn=net_loss, optimizer=opt)
    model.train(1, ds_train, callbacks=[loss_cb], dataset_sink_mode=True)