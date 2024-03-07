from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore.train import Model
import mindspore.nn as nn
import os
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore import context, Tensor
import logging
from src.finetune_config import cfg, bert_net_cfg
from src.poetry_dataset import create_poetry_dataset, create_tokenizer
from src.utils import BertPoetry, BertPoetryCell, BertLearningRate, BertPoetryModel
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
# context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

def train_net(model, epoch_size, dataset):
    print("============== Starting Training ==============")
    # Load training dataset
    
    
    # Define LossMonitor callback
    loss_cb = LossMonitor()
    # Start training
    model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=False)

from mindspore import export
from mindspore.nn.layer.basic import Dense

if __name__ == "__main__":
    epoch_size = 1
    repeat_size = epoch_size


    num_tokens = 3191
    poetrymodel = BertPoetryModel(bert_net_cfg, True, num_tokens, dropout_prob=0.1)
    poetry, tokenizer, keep_words = create_tokenizer()
    dataset = create_poetry_dataset(bert_net_cfg.batch_size, poetry, tokenizer)
    steps_per_epoch = dataset.get_dataset_size()
    lr_schedule = BertLearningRate(learning_rate=cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=1000,
                                       decay_steps=epoch_size*steps_per_epoch,
                                       power=cfg.AdamWeightDecay.power)
    netwithloss = BertPoetry(poetrymodel, bert_net_cfg, True, dropout_prob=0.1)
    optimizer = AdamWeightDecay(netwithloss.trainable_params(), lr_schedule)
    netwithloss = BertPoetry(poetrymodel, bert_net_cfg, True, dropout_prob=0.1)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = BertPoetryCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    train_net(model, epoch_size, repeat_size)