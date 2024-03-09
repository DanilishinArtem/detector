import os
import argparse
from src.utils import BertPoetry, BertPoetryCell, BertLearningRate, BertPoetryModel
from src.finetune_config import cfg, bert_net_cfg
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import Callback
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.poetry_dataset import create_poetry_dataset, create_tokenizer
from generator import generate_random_poetry, generate_hidden
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.train.serialization import export
import numpy as np
import time
import re
from mindspore.train.callback import LossMonitor
from src.bert_model import BertModel
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
import mindspore.nn as nn
from mindspore.nn.metrics import Accuracy


context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

def prepare():
    epoch_size = 1
    poetry, tokenizer, keep_words = create_tokenizer()

    dataset = create_poetry_dataset(bert_net_cfg.batch_size, poetry, tokenizer)
    loss_cb = LossMonitor()
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    lr = 0.01
    momentum = 0.9
    num_tokens = 3191
    poetrymodel = BertPoetryModel(bert_net_cfg, True, num_tokens, dropout_prob=0.1)
    model = BertPoetry(poetrymodel, bert_net_cfg, True, dropout_prob=0.1)
    
    model = Model(model)
    model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=False)

if __name__ == '__main__':
    prepare()
    
