"""蒙特卡洛树搜索"""
import numpy as np
import copy
from mcts_node import MctsNode
from engine.observation import Observation
from typing import List, Tuple
import os
import time


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# 蒙特卡洛搜索树
class MCTS(object):

    """policy_value_fn: (Observation -> *(Action, Prob), value)"""

    def __init__(self, policy_value_fn, n_playout=500):
        self.root = MctsNode(None, 1.0)
        self.policy_value_fn = policy_value_fn
        self.n_playout = n_playout
        self.temp = 1e-3
        self.t1 = self.t2 = self.t3 = self.t4 = self.t5 = self.t6 = self.t7 = 0
    
    def clear(self):
        self.t1 = 0 # select
        self.t2 = 0 # act
        self.t3 = 0 # network
        self.t4 = 0 # expand
        self.t5 = 0 # total
        self.t6 = 0
        self.t7 = 0
    
    def show_time(self):
        print("select %f, act: %f, network: %f, expand: %f, update: %f, status: %f, total: %f"\
            % (self.t1, self.t2, self.t3, self.t4, self.t6, self.t7, self.t5))

    def __str__(self):
        return ""

    def _helper(node: MctsNode, level: int):
        if node is None:
            return
        for i in range(level):
            print(" " * i * 4, end="")
        print()
        
    def show(self):
        node = self.root
        self._helper(node)

    def _playout(self, state: Observation):
        """
        进行一次rollout操作：即执行一次完整的游戏
        根据叶节点的评估值进行反向更新树节点的参数
        注意：state已就地修改，因此必须提供副本
        """
        t0 = time.time()
        flag1: bool = self.root.is_leaf()
        node = self.root
        while True:
            if node.is_leaf():
                break
            # 贪心算法选择下一步行动
            t1 = time.time()
            action, node = node.select()
            t2 = time.time()
            # 这里available_actions是会发生变化的, obs的step函数需要注意
            state.step(action)
            t3 = time.time()
            self.t1 = self.t1 + t2 - t1
            self.t2 = self.t2 + t3 - t2

        t70 = time.time()
        state.get_status()
        self.t7 = self.t7 + time.time() - t70
        # 查看游戏是否结束
        _is_terminated, winner = state.is_terminated()
        if not _is_terminated:
            # 使用网络评估叶子节点，网络输出（动作，概率）元组p的列表以及当前玩家视角的得分[-1, 1]
            t1 = time.time()
            action_probs, leaf_value = self.policy_value_fn(state)
            t2 = time.time()
            node.expand(action_probs)
            t3 = time.time()
            self.t3 = self.t3 + t2 - t1
            self.t4 = self.t4 + t3 - t2
            # if flag1 is True and len(list(action_probs)) == 0:
            #     print("ERROR: root is leaf but no next action.")
            #     for action in state.available_actions:
            #         print(action)
        else:  # 对于结束状态，将叶子节点的值换成1或-1
            if winner == -1:  # Tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.current_player_id() else -1.0
            print("leaf value: ", leaf_value)
        # 在本次遍历中更新节点的值和访问次数
        # 必须添加符号，因为两个玩家共用一个搜索树
        t6 = time.time()
        node.update_all_node_in_path(-leaf_value)
        t7 = time.time()
        self.t6 = self.t6 + t7 - t6
        self.t5 = self.t5 + t7 - t0

    def get_action_with_probs(self, obs: Observation) -> (Tuple, Tuple):
        """
        按顺序运行所有搜索并返回可用的动作及其相应的概率
        state:当前游戏的状态
        self.temp:介于（0， 1]之间的温度参数
        """
        # time_st = time.time()
        self.clear()
        for n in range(self.n_playout):
            obs_copy = copy.deepcopy(obs)
            self._playout(obs_copy)
        # time_rollout = time.time()
        # print("rollout time cost: ", time_rollout - time_st)
        self.show_time()

        # if len(self.root.children.items()) == 0:
        #     print("Error: no next action.")
        #     obs.show()
        #     return tuple(), tuple()
        # 跟据根节点处的访问计数来计算移动概率
        act_visits = [(act, node.n)
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        # print(type(acts), type(visits))
        act_probs = softmax(1.0 / self.temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        根据last_move，向搜索树的叶子节点方向移动搜索树的根节点
        如果是第一步，则新建一个根节点
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = MctsNode(None, 1.0)
