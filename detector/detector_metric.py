import torch
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, net, weights_flag, level):
        self.net = net
        self.fault = torch.tensor([0], dtype=torch.float)
        self.upper_metric = torch.tensor([0], dtype=torch.float)
        self.lower_metric = torch.tensor([0], dtype=torch.float)
        self.level = level
        self.metric_head = torch.tensor([0], dtype=torch.float)
        self.metric_sd = torch.tensor([0], dtype=torch.float)
        self.metric_history = torch.tensor([0], dtype=torch.float)
        self.const_detection = False
        self.prev = 0
        self.alpha = 0.9

    def detect(self, t, metric):
            t = t + 1
            self.metric_history[t-1] = metric
            if t == 1:
                self.metric_head[t-1] = metric
                self.metric_sd[t-1] = 0.001
                self.prev
            else:
                self.metric_sd[t-1] = torch.sqrt((((t-1)/t)*self.metric_sd[t-2]**2) + ((1/t)*(metric - self.metric_head[t-2])**2))
                self.metric_head[t-1] = self.metric_head[t-2] * ((t-1)/t) + (metric/t)
                # self.metric_sd[t-1] = torch.sqrt(((self.alpha)*self.metric_sd[t-2]**2) + ((1 - self.alpha)*(metric - self.metric_head[t-2])**2))
                # self.metric_head[t-1] = self.metric_head[t-2] * (self.alpha) + (metric *(1 - self.alpha))
                if metric == self.prev:
                    self.const_detection = True
                    self.prev = metric
            
            self.upper_metric[t-1] = self.metric_head[t-1] + self.level * self.metric_sd[t-1]
            self.lower_metric[t-1] = self.metric_head[t-1] - self.level * self.metric_sd[t-1]

            self.upper_metric = torch.cat((self.upper_metric, torch.tensor([0], dtype=torch.float)))
            self.lower_metric = torch.cat((self.lower_metric, torch.tensor([0], dtype=torch.float)))
            self.metric_head = torch.cat((self.metric_head, torch.tensor([0], dtype=torch.float)))
            self.metric_sd = torch.cat((self.metric_sd, torch.tensor([0], dtype=torch.float)))
            self.metric_history = torch.cat((self.metric_history, torch.tensor([0], dtype=torch.float)))
            self.fault = torch.cat((self.fault, torch.tensor([0], dtype=torch.float)))
            if metric > self.upper_metric[t-1] or metric < self.lower_metric[t-1] or self.const_detection:
                self.fault[t-1] = 1
                return True
            else:
                self.fault[t-1] = 0
                return False

    def show_statistics(self, plot_flag):
        num_faults = self.fault[0:-1].sum().item()
        total_num = len(self.fault[0:-1])
        print("number of faults: " + str(num_faults))
        print("total number of steps: " + str(total_num))
        print("faults (%): " + str(num_faults/total_num*100))
        if plot_flag:
            plt.plot(self.metric_history[0:-1].detach(), label="metric")
            plt.plot(self.metric_head[0:-1].detach(), label="head")
            plt.plot(self.upper_metric[0:-1].detach(), label="upper")
            plt.plot(self.lower_metric[0:-1].detach(), label="lower")
            plt.legend()
            plt.show()
