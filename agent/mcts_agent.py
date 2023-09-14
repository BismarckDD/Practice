import sys
sys.path.append("..")
from engine.action_hepler import id_2_action, action_2_id
from engine.action import Action
from engine.observation import Observation
from agent.base_agent import BaseAgent
from algorithm.config import CONFIG
from algorithm.mcts import MCTS
import numpy as np
from typing import Dict
import time


# 基于策略价值网络的MCTS-Agent
class MctsAgent(BaseAgent):

    def __init__(self, policy_value_function, n_playout=100, description="AI Based on MCTS"):
        self.mcts = MCTS(policy_value_function, n_playout)
        self.description = description
        self.n_playout = n_playout

    def __str__(self):
        return "MCTS: {}".format(self.description)

    # 重置搜索树，上一步操作是-1
    def reset(self):
        # super().reset()
        self.mcts.update_with_move(-1)

    def get_action(self, obs: Observation) -> (Action, Dict[int, float]):
        action_probs = np.zeros(2086)
        action_ids, probs = self.mcts.get_action_with_probs(obs)
        action_probs[list(action_ids)] = probs
        # 使用默认的temp=1e-3，它几乎相当于选择具有最高概率的移动
        action_id = np.random.choice(action_ids, p=probs)
        # 重置根节点
        self.mcts.update_with_move(-1)
        return id_2_action[action_id], action_probs

    def get_action_in_training(self, obs: Observation) -> (Action, Dict[int, float]):
        # t1 = time.time()
        action_probs = np.zeros(2086)
        # action_ids 和 probs 都是 tuple
        action_ids, probs = self.mcts.get_action_with_probs(obs)
        # t2 = time.time()
        # print(type(action_ids), type(probs))
        # 这里是针对ndarray进行操作，而不是dict，dict不支持list的操作
        action_probs[list(action_ids)] = probs
        # 添加Dirichlet Noise进行探索（自我对弈需要）
        float_probs = np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
        choice_prob = 0.75 * probs + 0.25 * float_probs
        action_id = np.random.choice(action_ids, p=choice_prob)
        # 更新根节点并重用搜索树
        self.mcts.update_with_move(action_id)
        # t3 = time.time()
        # print("get action with probs cost: ", t2 - t1)
        # print("get action int training cost: ", t3 - t1)
        return id_2_action[action_id], action_probs
