import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time



class Alalizer:
    def __init__(self, model, div_perc):
        self.model = model
        self.div_perc = div_perc
        self.fault = False
        self.comp_weight = 0
        self.comp_weight_grad = 0
        self.success_weight = None
        self.success_weight_grad = None
        self.first = True
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
        if self.first == True or self.fault == False:
            params = self.get_parameters()
            temp_params = torch.cat((params[0], params[1]))
            start_dim = len(temp_params)
            factors = self.find_factors(len(temp_params))
            x = int(factors[int(len(factors)/2)])
            y = int(len(temp_params) / x)
            temp_params = temp_params.reshape(x, y)
            pca = PCA(n_components=y)
            temp_params = pca.fit_transform(temp_params)
            cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
            index_99_variance = max(np.argmax(cumulative_variance_ratio >= self.div_perc), 1)
            temp_params = temp_params[:, :index_99_variance].flatten()
            transform_dim = len(temp_params)
            print('compression: ' + str((1 - (transform_dim / start_dim))*100) + '%')

        plt.figure(figsize=(20, 10))
        plt.hist(temp_params, bins=100, alpha=0.5, label='weights_and_grads')
        plt.title('weights_and_grad')
        plt.savefig("/home/adanilishin/detector/pictures/" + name + ".png")
        plt.clf()
        plt.close()



