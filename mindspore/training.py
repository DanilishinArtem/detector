from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore.train import Model
from model import LeNet5
import mindspore.nn as nn
from create_dataset import create_dataset
import os
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore import context, Tensor
import logging


# context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

def train_net(model, epoch_size, mnist_path, repeat_size):
    print("============== Starting Training ==============")
    # Load training dataset
    ds_train = create_dataset(os.path.join(mnist_path, "train"), 32, repeat_size)
    
    # Define LossMonitor callback
    loss_cb = LossMonitor()
    
    # Start training
    model.train(epoch_size, ds_train, callbacks=[loss_cb], dataset_sink_mode=False) 

from mindspore import export

if __name__ == "__main__":
    epoch_size = 1
    mnist_path = "/home/adanilishin/MNIST_Data"
    repeat_size = epoch_size
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    lr = 0.01
    momentum = 0.9
    network = LeNet5()
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    train_net(model, epoch_size, mnist_path, repeat_size)