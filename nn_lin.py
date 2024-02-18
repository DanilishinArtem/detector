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
        self.layer2 = nn.Linear(20, 20)  # Изменил размерность входа на 20
        self.layer2R = nn.ReLU()
        self.layer3 = nn.Linear(20, 20)  # Изменил размерность входа на 20
        self.layer3R = nn.ReLU()
        self.layer4 = nn.Linear(20, 20)  # Изменил размерность входа на 20
        self.layer4R = nn.ReLU()
        self.layer5 = nn.Linear(20, 20)  # Изменил размерность входа на 20
        self.layer5R = nn.ReLU()
        self.layer6 = nn.Linear(20, 20)  # Изменил размерность входа на 20
        self.layer6R = nn.ReLU()
        self.layer7 = nn.Linear(20, 20)  # Изменил размерность входа на 20
        self.layer7R = nn.ReLU()
        self.layer8 = nn.Linear(20, 1)  # Изменил размерность входа на 20
        
    def forward(self, x):
        xOneOut1 = self.layer1R(self.layer1(x))
        xOneOut2 = self.layer2R(self.layer2(xOneOut1))
        xOneOut3 = self.layer3R(self.layer3(xOneOut2))
        xOneOut4 = self.layer4R(self.layer4(xOneOut3))
        xOneOut5 = self.layer5R(self.layer5(xOneOut4))
        xOneOut6 = self.layer6R(self.layer6(xOneOut5))
        xOneOut7 = self.layer7R(self.layer7(xOneOut6))
        out = self.layer8(xOneOut7)
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
    # analizer.disp_hist("epoch_" + str(i))

# detector.show_statistics(True)



