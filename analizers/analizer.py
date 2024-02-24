import numpy as np
import torch
import matplotlib.pyplot as plt
import time


class Alalizer:
    def __init__(self, model):
        self.model = model
        self.first = True
    def get_parameters(self):
        weights = torch.tensor([])
        weights_grad = torch.tensor([])
        layers = list(self.model.named_parameters())
        for i in range(len(layers)):
            layer = layers[i][0]
            list_of_attr = layer.split(".")
            current_attr = self.model
            for attr in list_of_attr:
                current_attr = getattr(current_attr, attr)
                if attr == 'weight' or attr == 'bias':
                    weights = torch.cat((weights, torch.flatten(current_attr.data)))
                    weights_grad = torch.cat((weights, torch.flatten(current_attr.grad.data)))
        return (weights, weights_grad)
    def disp_hist(self, name):
        params = self.get_parameters()
        if self.first:
            plt.figure(figsize=(20, 10))
            self.first = False
        plt.subplot(1,2,1)
        plt.hist(params[0], bins=100, alpha=0.5, label='weights')
        plt.title('weights')
        plt.subplot(1,2,2)
        plt.hist(params[1], bins=100, alpha=0.5, label='weights_grad')
        plt.title('weights_grad')
        plt.show(block=False)
        plt.savefig("/home/adanilishin/work/detector/pictures/" + name + ".png")
        # time.sleep(2)
        plt.clf()
        # plt.close()