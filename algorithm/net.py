import numpy as np
import sys
sys.path.append("..")
from resnet_block import ResidualBlock
from resnet_block import ConvBlock
from resnet_block import PolicyHeader
from resnet_block import ValueHeader
from torchvision import datasets, transforms
import torch.optim as opt
import torch.nn as nn
import torch
from engine.observation import Observation
from engine.action import Action


class Model(nn.Module):

    def __init__(self, channels=256, num_res_blocks=7):
        super().__init__()
        self.conv_block = ConvBlock()
        self.res_blocks = nn.Sequential(*[ResidualBlock() for _ in range(num_res_blocks)])
        self.policy_header = PolicyHeader()
        self.value_header = ValueHeader()

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)
        policy = self.policy_header(x)
        value = self.value_header(x)
        return policy, value


class AverageNetwork(object):
    def __init__(self):
        super().__init__(self)

    def policy_value_fn(self, observation: Observation) -> ({}, float):
        actions = observation.available_actions
        l = len(actions)
        if l == 0:
            return {}, 0.0
        probs = [1 / l] * l
        return zip(actions, probs), 1.0



class PolicyValueNetworkTest:

    def __init__(self):
        self.model = Model()
        self.l2_const = 2e-3  # 正则化
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = opt.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self, max_epoches: int, train_loader: torch.utils.data.DataLoader):
        for epoch in range(max_epoches):
            running_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

    def inference(self, state_batch):
        self.model.eval()
        state_batch = torch.tensor(state_batch)
        log_act_probs, value = self.model(state_batch)
        act_probs = np.exp(log_act_probs)
        return act_probs, value.numpy()

    def policy_value_fn(self, observation: Observation) -> ({}, float):
        self.model.eval()
        # 获取合法的动作列表
        board = observation.np_board
        legal_actions = observation.available_actions()
        # 连续内存数组，会提升性能
        current_state = np.ascontiguousarray(board.reshpae(-1, 9, 10, 9)).astype('float32')
        current_state = torch.tensor(current_state)
        log_action_probs, value = self.model(current_state)
        action_probs = np.exp(log_action_probs).numpy().flatten()
        action_probs = zip(legal_actions, action_probs(legal_actions))
        return action_probs, value.numpy()


if __name__ == '__main__':
    # net = Model()
    # NUM, CHANNELS, HEIGHT, WIDTH
    # test_data = torch.ones([8, 1, 10, 9])
    # x_act, x_val = net.forward(test_data)
    # print(x_act, x_act.shape)
    # print(x_val, x_val.shape)

    net = PolicyValueNetwork()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # train_set = datasets.MNIST("D:\\workspace\\ChineseChess\\", download=True, train=True, transform=transform)
    # data_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    # net.train(2000, data_loader)


