from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore.train import Model
from model import LeNet5
import mindspore.nn as nn
from create_dataset import create_dataset
import os
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits


def train_net(model, epoch_size, mnist_path, repeat_size):
    print("============== Starting Training ==============")
    #load training dataset
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    ds_train = create_dataset(os.path.join(mnist_path, "train"), 32, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=False) 


if __name__ == "__main__":
    epoch_size = 1
    mnist_path = "./MNIST_Data"
    repeat_size = epoch_size
    net_loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    lr = 0.01
    momentum = 0.9
    network = LeNet5()
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    train_net(model, epoch_size, mnist_path, repeat_size)