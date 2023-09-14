import torch
import torch.nn as nn
import torch.nn.functional as func


class ResidualBlock(nn.Module):

    def __init__(self, input_channels=256, output_channels=256):
        super(ResidualBlock, self).__init__()
        # 每个conv都有独立的参数权重，因此要分开写
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=output_channels)
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=output_channels)

    def forward(self, param):
        output = self.conv1(param)
        output = self.batch_norm1(output)
        output = func.relu(output)
        output = self.conv2(output)
        output = self.batch_norm2(output)
        output = param + output
        output = func.relu(output)
        return output


class ConvBlock(nn.Module):

    def __init__(self, input_channels=9, output_channels=256):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=output_channels)

    def forward(self, param):
        output = self.conv(param)
        output = self.batch_norm(output)
        output = func.relu(output)
        return output


class ValueHeader(nn.Module):

    def __init__(self, input_channels=256, output_channels=8):
        super(ValueHeader, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1,
                              padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=output_channels)
        self.fc = nn.Linear(8 * 9 * 10, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, param):
        output = self.conv(param)
        output = self.batch_norm(output)
        output = func.relu(output)
        output = torch.reshape(output, [-1, 8 * 10 * 9])
        output = self.fc(output)
        output = self.fc2(output)
        return output


class PolicyHeader(nn.Module):

    def __init__(self, input_channels=256, output_channels=16):
        super(PolicyHeader, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1,
                              padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=output_channels)
        self.fc = nn.Linear(16 * 9 * 10, 2086)

    def forward(self, param):
        output = self.conv(param)
        output = self.batch_norm(output)
        output = func.relu(output)
        output = torch.reshape(output, [-1, 16 * 10 * 9])
        output = self.fc(output)
        # log_softmax 和 softmax 的区别
        output = func.log_softmax(output)
        return output


