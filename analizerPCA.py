import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
def LD(n):
    return max([i for i in range(1, n) if n % i == 0], default=1)

class AlalizerPCA:
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
                    weights = torch.cat((weights, torch.flatten(current_attr.data.to('cpu'))))
                    weights_grad = torch.cat((weights_grad, torch.flatten(current_attr.grad.data.data.to('cpu'))))
                elif attr == 'bias':
                    bias = torch.cat((bias, torch.flatten(current_attr.data.data.to('cpu'))))
                    bias_grad = torch.cat((bias_grad, torch.flatten(current_attr.grad.data.to('cpu'))))
        return (bias, weights, bias_grad, weights_grad)
    def disp_hist(self):
        if self.first:
            plt.figure(figsize=(10, 5))
            self.first = False
        params = self.get_parameters()
        nComponents_bias = int(LD(len(params[0])))
        nComponents_weights = int(LD(len(params[1])))
        bias_matrix = params[0].view(nComponents_bias, -1)
        weights_matrix = params[1].view(nComponents_weights, -1)
        bias_grad_matrix = params[2].view(nComponents_bias, -1)
        weights_grad_matrix = params[3].view(nComponents_weights, -1)
        del params
        pca = PCA(n_components=1)
        bias_pca = pca.fit_transform(bias_matrix)
        weights_pca = pca.fit_transform(weights_matrix)
        bias_grad_pca = pca.fit_transform(bias_grad_matrix)
        weights_grad_pca = pca.fit_transform(weights_grad_matrix)

        plt.subplot(2,2,1)
        plt.hist(bias_pca, bins=100, alpha=0.5, label='bias')
        plt.title('bias')
        plt.subplot(2,2,2)
        plt.hist(weights_pca, bins=100, alpha=0.5, label='weights')
        plt.title('weights')
        plt.subplot(2,2,3)
        plt.hist(bias_grad_pca, bins=100, alpha=0.5, label='bias_grad')
        plt.title('bias_grad')
        plt.subplot(2,2,4)
        plt.hist(weights_grad_pca, bins=100, alpha=0.5, label='weights_grad')
        plt.title('weights_grad')
        plt.show(block=False)
        plt.pause(1)
        plt.clf()