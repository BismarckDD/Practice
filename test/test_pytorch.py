import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

train = datasets.MNIST('data/', download=True, train=True)
test = datasets.MNIST('data/', download=True, train=False)

X_train = train.data.unsqueeze(1) / 255.0
y_train = train.targets
trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

X_test = test.data.unsqueeze(1) / 255.0
y_test = test.targets


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(len(x), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = LeNet5()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        for X, y in trainloader:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            y_pred = model(X_train)
            acc_train = (y_pred.argmax(dim=1) == y_train).float().mean().item()
            y_pred = model(X_test)
            acc_test = (y_pred.argmax(dim=1) == y_test).float().mean().item()
            print(epoch, acc_train, acc_test)
