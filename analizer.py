import numpy as np
import torch
import matplotlib.pyplot as plt
import time


class Alalizer:
    def __init__(self, model):
        self.model = model
    def get_parameters(self):
        bias = torch.tensor([])
        weights = torch.tensor([])
        layers = list(self.model.named_parameters())
        for i in range(len(layers)):
            layer = layers[i][0]
            list_of_attr = layer.split(".")
            current_attr = self.model
            for attr in list_of_attr:
                current_attr = getattr(current_attr, attr)
                if attr == 'weight':
                    weights = torch.cat((weights, torch.flatten(current_attr.data)))
                elif attr == 'bias':
                    bias = torch.cat((bias, torch.flatten(current_attr.data)))
        return (bias, weights)
    def disp_hist(self, name):
        params = self.get_parameters()
        # plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.hist(params[0], bins=300, alpha=0.5, label='bias')
        plt.title('bias')
        plt.subplot(1,2,2)
        plt.hist(params[1], bins=300, alpha=0.5, label='weights')
        plt.title('weights')
        plt.show(block=False)
        plt.savefig("./pictures/" + name + ".png")
        # time.sleep(2)
        # plt.clf()
        plt.close()