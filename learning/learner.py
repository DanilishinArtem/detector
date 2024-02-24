import torch


class Learner:
    def __init__(self, model, trainloader, testloader, epochs, criterion, optimizer):
        self.model = model
        self.epochs = epochs
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
    def run_learning(self):
        for epoch in range(self.epochs):  # Проходим по данным несколько раз
            running_loss = 0.0
            counter = 0
            for data in self.trainloader:
                counter += 1
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print('loss function for epoch ' + str(epoch) + ' = ' + str(running_loss / counter))
            running_loss = 0
        print('Обучение закончено')
    def run_testing(self):
        correct = 0
        total = 0
        counter = 0
        with torch.no_grad():
            for data in self.testloader:
                counter += 1
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('accuracy during ' + str(counter) + ' numbers = ' + str(100 * correct / total) + '%')
