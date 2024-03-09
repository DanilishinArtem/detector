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


class LossCallBack(Callback):
    '''
    Monitor the loss in training.
    If the loss is NAN or INF, terminate training.
    Note:
        If per_print_times is 0, do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    '''
    def __init__(self, model, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be in and >= 0.")
        self._per_print_times = per_print_times
        self.model = model

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        with open("./loss.log", "a+") as f:
            f.write("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                                 str(cb_params.net_outputs)))
            f.write("\n")


def prepare():
    epoch_size = 1
    poetry, tokenizer, keep_words = create_tokenizer()

    dataset = create_poetry_dataset(bert_net_cfg.batch_size, poetry, tokenizer)
    
    num_tokens = 3191
    poetrymodel = BertPoetryModel(bert_net_cfg, True, num_tokens, dropout_prob=0.1)
    netwithloss = BertPoetry(poetrymodel, bert_net_cfg, True, dropout_prob=0.1)
    callback = LossCallBack(poetrymodel)

    steps_per_epoch = dataset.get_dataset_size()
    lr_schedule = BertLearningRate(learning_rate=cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=1000,
                                       decay_steps=cfg.epoch_num*steps_per_epoch,
                                       power=cfg.AdamWeightDecay.power)
    optimizer = AdamWeightDecay(netwithloss.trainable_params(), lr_schedule)
    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix=cfg.ckpt_prefix, directory=cfg.ckpt_dir, config=ckpt_config)

    param_dict = load_checkpoint(cfg.pre_training_ckpt)
    new_dict = {}

    for key in param_dict:
        if "bert_embedding_lookup" not in key:
            new_dict[key] = param_dict[key]
        else:
            value = param_dict[key]
            np_value = value.data.asnumpy()
            np_value = np_value[keep_words]
            tensor_value = Tensor(np_value, mstype.float32)
            parameter_value = Parameter(tensor_value, name=key)
            new_dict[key] = parameter_value

    load_param_into_net(netwithloss, new_dict)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = BertPoetryCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)

    model = Model(netwithgrads)

    loss_cb = LossMonitor()
    # model.train(cfg.epoch_num, dataset, callbacks=[loss_cb], dataset_sink_mode=True)
    model.train(cfg.epoch_num, dataset, callbacks=[callback, ckpoint_cb], dataset_sink_mode=False)

if __name__ == '__main__':
    prepare()
    
