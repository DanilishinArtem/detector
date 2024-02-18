import torch
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, net, weights_flag, confInt):
        self.net = net
        self.confInt = confInt
        self.weights_flag = weights_flag
        self.bias = torch.tensor([0], dtype=torch.float)
        self.bias_sd = torch.tensor([0], dtype=torch.float)
        self.bias_fault = torch.tensor([0], dtype=torch.float)

        self.weights = torch.tensor([0], dtype=torch.float)
        self.weights_sd = torch.tensor([0], dtype=torch.float)
        self.weights_fault = torch.tensor([0], dtype=torch.float)

        self.alpha = 0.9
        self.beta = 0.99
    def detect(self, t, metric):
        current_bias, current_weights = self.get_means()
        if t == 0:
            self.bias[t] = current_bias
            self.bias_sd[t] = 0.0001

            self.weights[t] = current_weights
            self.weights_sd[t] = 0.0001
        else:
            self.bias_sd[t] = self.beta * (self.bias_sd[t - 1] ** 2) + (1 - self.beta) * ((current_bias - self.bias[t - 1]) ** 2)
            self.bias[t] = self.alpha * self.bias[t - 1] + (1 - self.alpha) * current_bias

            self.weights_sd[t] = self.beta * (self.weights_sd[t - 1] ** 2) + (1 - self.beta) * ((current_weights - self.weights[t - 1]) ** 2)
            self.weights[t] = self.alpha * self.weights[t - 1] + (1 - self.alpha) * current_weights

        bias_lower = self.bias[t] - self.confInt * self.bias_sd[t]
        bias_upper = self.bias[t] + self.confInt * self.bias_sd[t]

        weights_lower = self.weights[t] - self.confInt * self.weights_sd[t]
        weights_upper = self.weights[t] + self.confInt * self.weights_sd[t]

        if self.bias[t] <= bias_lower:# or self.bias[t] >= bias_upper:
            self.bias_fault[t] = 1
        else:
            self.bias_fault[t] = 0

        if self.weights[t] <= weights_lower:# or self.weights[t] >= weights_upper:
            self.weights_fault[t] = 1
        else:
            self.weights_fault[t] = 0

        self.bias = torch.cat((self.bias, torch.tensor([0])))
        self.bias_sd = torch.cat((self.bias_sd, torch.tensor([0])))
        self.weights = torch.cat((self.weights, torch.tensor([0])))
        self.weights_sd = torch.cat((self.weights_sd, torch.tensor([0])))

        self.bias_fault = torch.cat((self.bias_fault, torch.tensor([0])))
        self.weights_fault = torch.cat((self.weights_fault, torch.tensor([0])))
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
        return bias, weights
    def get_means(self):
        bias, weights = self.get_params()
        return bias.mean(), weights.mean()
    def show_statistics(self, plot_flag):
        if plot_flag:
            plt.subplot(1,2,1)
            plt.plot(self.bias[0:-1].detach(), label="bias")
            plt.plot(self.bias[0:-1] + self.confInt * self.bias_sd[0:-1], label="upper")
            # plt.plot(self.bias[0:-1] - self.confInt * self.bias_sd[0:-1], label="lower")
            # plt.plot(self.bias_fault[0:-1], label="fault")
            plt.title("bias")
            plt.legend()

            plt.subplot(1,2,2)
            plt.plot(self.weights[0:-1].detach(), label="weight")
            plt.plot(self.weights[0:-1] + self.confInt * self.weights_sd[0:-1], label="upper")
            # plt.plot(self.weights[0:-1] - self.confInt * self.weights_sd[0:-1], label="lower")
            # plt.plot(self.weights_fault[0:-1], label="fault")
            plt.title("weight")
            plt.legend()

            plt.show()

        bias_num = self.bias_fault[0:-1].sum().item()
        weights_num = self.weights_fault[0:-1].sum().item()
        total_len = len(self.bias_fault[0:-1])

        print("number of faults (bias): " + str(bias_num))
        print("number of faults: (weights): " + str(weights_num))
        print("total number of faults: " + str(bias_num + weights_num))
        print("total len: " + str(total_len))
        total_faults = ((self.bias_fault.detach() > 0) | (self.weights_fault.detach() > 0)).sum().item()
        print("faults (%): " + str((total_faults) / total_len * 100))

