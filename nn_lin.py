import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import matplotlib.pyplot as plt
import torch
from hooks import GradHook
from detector_metric import Detector
from analizer import Alalizer
# from detector_article import Detector
# from detector_sd_mean import Detector
# from detector_sd_each import Detector


def getDatas(n, showDatas=False):
    x = np.random.rand(n, 1)
    y = 3 * (x ** 3) + (x ** 2) + 0.5
    # transform to tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    if showDatas:
        plt.plot(y)
        plt.show()
    return x, y
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(1, 20)
        self.layer1R = nn.ReLU()
        self.layer2 = nn.Linear(20, 1)
        self.layer2.register_backward_hook(GradHook(1).hook)
        # self.layer2R = nn.ReLU()
    def forward(self, x):
        xOneOut = self.layer1R(self.layer1(x))
        out = self.layer2(xOneOut)
        return out

net = Network()
detector = Detector(net, False, 2)

x, y = getDatas(1000, False)
epochs = 20

analizer = Alalizer(net)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
for i in range(epochs):
    analizer.disp_hist("epoch_" + str(i))
    train_loss = 0
    for j in range(len(x)):
        optimizer.zero_grad()
        yMod = net(x[j:(j + 1)])
        loss = criterion(yMod, y[j:(j + 1)])
        loss.backward()
        # detector.detect((len(x)*i)+j, loss)

        optimizer.step()
        train_loss += loss.item()
    print("loss for epoch " + str(i) + " = " + str(train_loss))

# detector.show_statistics(True)



