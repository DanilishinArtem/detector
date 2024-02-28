import mindspore.nn.cell as nn
import logging
from my_hooks import create_forward_hook


model = None

backup = nn.Cell.__init__
def override(self, auto_prefix=True, flags=None):
    global model
    backup(self, auto_prefix=True, flags=None)
    if 'model' in self._cell_tag:
        model = self
        logging.warning('[Model getted]: ' + str(model._cell_tag))
    elif 'mindspore.nn.optim' in self._cell_tag:
        for name, layer in model._cells.items():
            if name == 'fc2':
                # logging.warning('[forward hook implemented for layer ' + name + ']:')
                # hook = create_forward_hook(1000, 1, 20, layer.weight, layer.bias)
                # layer.register_forward_hook(hook)

                logging.warning('[backward hook implemented for layer ' + name + ']:')
                hook = create_forward_hook(1000, 1, 20, layer.weight, layer.bias)
                layer.register_backward_hook(hook)
nn.Cell.__init__ = override
