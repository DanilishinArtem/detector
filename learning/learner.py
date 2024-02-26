import torch
# from analizers.analizer import Alalizer
# from analizers.analizerPCA import Alalizer
from analizers.analizerPCA_test import Alalizer

class Learner:
    def __init__(self, model, trainloader, testloader, epochs, criterion, optimizer, device):
        self.model = model
        self.epochs = epochs
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.analizer = Alalizer(model, 0.99)
        # self.analizer = Alalizer(model)
    def run_learning(self):
        self.model = self.model.to(self.device)
        for epoch in range(self.epochs):
            running_loss = 0.0
            counter = 0
            for data in self.trainloader:
                counter += 1
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if counter % 1000 == 0 or counter == 1:
                    self.analizer.disp_hist('epoch_' + str(epoch) + '_counter_' + str(counter))
                    print('loss function for number ' + str(counter) + ' = ' + str(running_loss / counter))
            print('loss function for epoch ' + str(epoch) + ' = ' + str(running_loss / counter))
            running_loss = 0
            print('total number of samples = ' + str(counter))
        self.model = self.model.to(self.device)
        print('Обучение закончено')
    def run_testing(self):
        correct = 0
        total = 0
        counter = 0
        self.model = self.model.to(self.device)
        with torch.no_grad():
            for data in self.testloader:
                counter += 1
                images, labels = data
                self.images = self.images.to(self.device)
                self.labels = self.labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('accuracy during ' + str(counter) + ' numbers = ' + str(100 * correct / total) + '%')
        self.model = self.model.to('cpu')
