import torch
import numpy as np
from typing import Tuple, List
from net import Model
from torch.cuda.amp import autocast
import torch.nn.functional as F
from engine.observation import Observation
from engine.action_hepler import generate_all_actions
import os


ONE = 1

# 策略值网络，用来进行模型的训练
class PolicyValueNet:

    def __init__(self, checkpoint_file=None, type="inference", device="cuda"):
        # set environ
        torch.set_num_threads(ONE)
        os.environ["OMP_NUM_THREADS"] = str(ONE)
        os.environ["MKL_NUM_THREADS"] = str(ONE)
        os.environ["OPENBLAS_NUM_THREADS"] = str(ONE)

        self.l2_const = 2e-3  # l2 正则化
        self.device = device
        # data_type should be defined as a member var
        # data_type should be decided by train or inference.
        self.data_type = torch.float16 if type=="inference" else torch.float32
        self.data_type_str = "float16" if type=="inference" else "float32"
        self.model = Model().to(device=self.device, dtype=self.data_type)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.l2_const)
        if checkpoint_file:
            self.model.load_state_dict(torch.load(checkpoint_file))  # 加载模型参数
        self.action_2_id, self.id_2_action = generate_all_actions()

    # 输入一个批次的状态，输出一个批次的动作概率和状态价值
    def policy_value(self, state_batch):
        self.model.eval()
        state_batch = torch.tensor(state_batch).to(self.device, dtype=self.data_type)
        log_act_probs, value = self.model(state_batch)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy())
        return act_probs, value.detach().numpy()

    # 输入棋盘，返回每个合法动作的（动作，概率）元组列表，以及棋盘状态的分数
    def policy_value_fn(self, obs: Observation) -> (List, float):
        self.model.eval()
        # 获取合法动作列表
        legal_actions = obs.available_actions
        legal_action_ids = list(map(lambda action: self.action_2_id[hash(action)], legal_actions))
        current_state = np.ascontiguousarray(obs.current_state().reshape(-1, 9, 9, 10))
        # current_state = current_state.astype(self.data_type)
        current_state = torch.as_tensor(current_state).to(device=self.device, dtype=self.data_type)
        # 使用神经网络进行预测
        with autocast():
            log_act_probs, value = self.model(current_state)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy().astype(self.data_type_str).flatten())
        # print(act_probs.shape, len(list(legal_action_ids)))
        # 只取出合法动作
        act_probs_list = list(zip(legal_action_ids, map(lambda key: act_probs[key], legal_action_ids)))
        if len(act_probs_list) != len(legal_action_ids):
            print("ERROR: Failed to extract legal action ids.")
        # 返回动作概率，以及状态价值
        return act_probs_list, value.detach().numpy()

    """
    " 保存模型checkpoint.
    """
    def save_model(self, model_file):
        torch.save(self.model.state_dict(), model_file)

    """
    " 执行一步训练操作
    " state_batch：当前状态 
    " action_probs: 由mcts计算得出的每个action的probs，并取log
    " winner_batch: 当前状态的估值(奖励)
    """
    def train_step(self, state_batch, action_probs, winner_batch, lr=0.002):
        self.model.train()
        # Create a tensor from ndarray. Put the tensor onto GPU.
        state_batch = torch.tensor(state_batch).to(self.device, dtype=self.data_type)
        action_probs = torch.tensor(action_probs).to(self.device, dtype=self.data_type)
        winner_batch = torch.tensor(winner_batch).to(self.device, dtype=self.data_type)
        # Clear grad in optimizer
        self.optimizer.zero_grad()
        # 设置学习率
        for params in self.optimizer.param_groups:
            # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9
            params['lr'] = lr
        # Forward op: get action_probs && value
        log_act_probs, value = self.model(state_batch)
        value = torch.reshape(value, shape=[-1])
        # 价值损失
        value_loss = F.mse_loss(input=value, target=winner_batch)
        # Policy Loss function.
        policy_loss = -torch.mean(torch.sum(action_probs * log_act_probs, dim=1))
        # 总的损失包括价值损失和策略损失，l2惩罚已经在优化器内部生效
        loss = value_loss + policy_loss
        # 反向传播及优化
        loss.backward()
        self.optimizer.step()
        # 计算策略的熵，仅用于评估模型
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()
