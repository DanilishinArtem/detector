import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from hooks import GradHook
from detector_metric import Detector
# from detector_article import Detector
# from detector_sd_mean import Detector
# from detector_sd_each import Detector

def plot_predict(model, x, y):
    with torch.no_grad():
        predicted_y = model(x)
    plt.scatter(x, y, label='Actual')
    plt.scatter(x, predicted_y, label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
# Генерация данных для бинарной классификации
def generate_binary_data(n, show_data=False):
    np.random.seed(42)
    x = np.random.rand(n, 1)
    y = (x > 0.5).astype(float)  # Простая бинарная классификация
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    if show_data:
        plt.scatter(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Binary Classification Data')
        plt.show()
    
    return x, y

# Нейронная сеть для бинарной классификации
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(1, 20)
        self.layer1_relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 1)
        self.layer2.register_backward_hook(GradHook(10).hook)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1_relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x


# Функция обучения модели
def train_binary_classifier(model, criterion, optimizer, x, y, epochs=10000):
    detector = Detector(model, False, 2)
    for epoch in range(epochs):
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # detector.detect(epoch, loss)
        detector.detect(epoch, loss)

        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    detector.show_statistics(True)

# Генерация данных
x, y = generate_binary_data(100, show_data=False)
# Создание модели
model = BinaryClassifier()
# Определение функции потерь и оптимизатора
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# Обучение модели
train_binary_classifier(model, criterion, optimizer, x, y)

plot_predict(model, x, y)