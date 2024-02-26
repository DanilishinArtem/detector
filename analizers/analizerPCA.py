import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time



class Alalizer:
    def __init__(self, model, div_perc):
        self.model = model
        self.div_perc = div_perc
    def find_factors(self, n):
        factors = []
        # Проверяем числа от 1 до n
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append(i)
        return factors
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
                    weights = torch.cat((weights, torch.flatten(current_attr.data.detach().to('cpu'))))
                    weights_grad = torch.cat((weights_grad, torch.flatten(current_attr.grad.data.detach().to('cpu'))))
        return [weights, weights_grad]
    def disp_hist(self, name):
        params = self.get_parameters()
        start_dim_weights = len(params[0])
        start_dim_weights_grad = len(params[1])
        factors = self.find_factors(len(params[0]))
        x = int(factors[int(len(factors)/2)])
        y = int(len(params[0]) / x)
        params[0] = params[0].reshape(x, y)
        params[1] = params[1].reshape(x, y)
        pca = PCA(n_components=y)

        params[0] = pca.fit_transform(params[0])
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        index_99_variance = max(np.argmax(cumulative_variance_ratio >= self.div_perc), 1)
        params[0] = params[0][:, :index_99_variance].flatten()

        params[1] = pca.fit_transform(params[1])
        cummulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        index_99_variance = max(np.argmax(cummulative_variance_ratio >= self.div_perc), 1)
        params[1] = params[1][:, :index_99_variance].flatten()

        print('compression for weights: ' + str((1 - (len(params[0]) / start_dim_weights))*100) + '%')
        print('compression for weights_grad: ' + str((1 - (len(params[1]) / start_dim_weights_grad))*100) + '%')

        plt.figure(figsize=(20, 10))
        plt.subplot(1,2,1)
        plt.hist(params[0], bins=100, alpha=0.5, label='weights')
        plt.title('weights')
        plt.subplot(1,2,2)
        plt.hist(params[1], bins=100, alpha=0.5, label='weights_grad')
        plt.title('weights_grad')
        plt.savefig("/home/adanilishin/detector/pictures/" + name + ".png")
        plt.clf()
        plt.close()