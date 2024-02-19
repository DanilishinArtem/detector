import numpy as np
import torch
import matplotlib.pyplot as plt
import time


class Alalizer:
    def __init__(self, model):
        self.model = model
        self.first = True
    def get_parameters(self):
        bias = torch.tensor([])
        weights = torch.tensor([])
        bias_grad = torch.tensor([])
        weights_grad = torch.tensor([])
        layers = list(self.model.named_parameters())
        for i in range(len(layers)):
            layer = layers[i][0]
            list_of_attr = layer.split(".")
            current_attr = self.model
            for attr in list_of_attr:
                current_attr = getattr(current_attr, attr)
                if attr == 'weight':
                    weights = torch.cat((weights, torch.flatten(current_attr.data)))
                    weights_grad = torch.cat((weights_grad, torch.flatten(current_attr.grad.data)))
                elif attr == 'bias':
                    bias = torch.cat((bias, torch.flatten(current_attr.data)))
                    bias_grad = torch.cat((bias_grad, torch.flatten(current_attr.grad.data)))
        return (bias, weights, bias_grad, weights_grad)
    def disp_hist(self):
        if self.first:
            plt.figure(figsize=(10, 5))
            self.first = False
        params = self.get_parameters()
        plt.subplot(2,2,1)
        plt.hist(params[0], bins=100, alpha=0.5, label='bias')
        plt.title('bias')
        plt.subplot(2,2,2)
        plt.hist(params[1], bins=100, alpha=0.5, label='weights')
        plt.title('weights')
        plt.subplot(2,2,3)
        plt.hist(params[2], bins=100, alpha=0.5, label='bias_grad')
        plt.title('bias_grad')
        plt.subplot(2,2,4)
        plt.hist(params[3], bins=100, alpha=0.5, label='weights_grad')
        plt.title('weights_grad')
        plt.show(block=False)
        plt.pause(1)
        plt.clf()