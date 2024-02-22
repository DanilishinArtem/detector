import torch
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, net, weights_flag, level):
        self.net = net
        self.weights_flag = weights_flag
        self.fault = torch.tensor([0], dtype=torch.float)
        self.alpha = 0.9
        self.beta = 0.99
        self.level = level
        self.params_head = 0
    def detect(self, t, metric):
        params = self.get_params()
        t = t + 1
        if t == 1:
            self.params_head = torch.zeros_like(params)
        else:
            self.params_head.mul_(self.alpha).add_(params, alpha=(1 - self.alpha))
        correction = 1 / (1 - (self.alpha ** t))
        self.params_head.mul_(correction)
        if torch.greater(self.params_head, self.level).any().item():
            self.fault[t-1] = 1
            self.fault = torch.cat((self.fault, torch.tensor([0], dtype=torch.float)))
            return True
        else:
            self.fault[t-1] = 0
            self.fault = torch.cat((self.fault, torch.tensor([0], dtype=torch.float)))
            return False
    def get_params(self):
        bias = torch.tensor([])
        weights = torch.tensor([])
        layers = list(self.net.named_parameters())
        for i in range(len(layers)):
            layer = layers[i][0]
            list_of_attr = layer.split(".")
            current_attr = self.net
            for attr in list_of_attr:
                current_attr = getattr(current_attr, attr)
                if self.weights_flag:
                    if attr == 'weight':
                        weights = torch.cat((weights, torch.flatten(current_attr.data)))
                    elif attr == 'bias':
                        bias = torch.cat((bias, torch.flatten(current_attr.data)))
                else:
                    if attr == 'weight':
                        weights = torch.cat((weights, torch.flatten(current_attr.grad.data)))
                    elif attr == 'bias':
                        bias = torch.cat((bias, torch.flatten(current_attr.grad.data)))
        return torch.cat((bias, weights))
    def show_statistics(self, plot_flag):
        num_faults = self.fault[0:-1].sum().item()
        total_num = len(self.fault[0:-1])
        print("number of faults: " + str(num_faults))
        print("total number of steps: " + str(total_num))
        print("faults (%): " + str(num_faults/total_num*100))
        if plot_flag:
            plt.plot(self.fault[0:-1], label="fault")
            plt.legend()
            plt.show()
