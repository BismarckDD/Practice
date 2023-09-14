import numpy as np
from config import CONFIG
from typing import Dict, Tuple


C_PUCT = CONFIG['c_puct']


# 定义MCTS的Node节点
class MctsNode(object):
    """
    mcts树中的节点，树的子节点字典中，键为动作，值为MctsNode。
    记录当前节点选择的动作，以及选择该动作后会跳转到的下一个子节点。
    每个节点跟踪其自身的Q，先验概率P及其访问次数调整的u
    """

    def __init__(self, parent, prior_p):
        self.parent: MctsNode = parent  # 当前节点的父节点
        self.children: Dict[int, int] = {}  # 当前节点的子节点
        self.n: int = 0  # 当前当前节点的访问次数
        self.u: float = 0  # 当前节点的置信上限 UCB
        self.Q: float = 0  # 当前节点对应动作的平均动作价值
        self.P: float = prior_p  # 选取当前节点的先验概率
        # self._value = 0
        # self.update_value()
    
    def expand(self, action_probs):
        """action prior 可以由神经网络构计算得出
        在没有成熟的神经网络时，也可以先随机给出"""
        p1, p2 = 0, 0  # p1 p2 is for debugging, not necessary
        for (action, prob) in action_probs:
            if action not in self.children.keys():
                self.children[action] = MctsNode(self, prob)
                p1 = p1 + 1
            else:
                p2 = p2 + 1
        # print(p1, p1 + p2, len(self.children.keys()))
        # print("self.children", self.children)

    def select(self):
        """
        在子节点中选择能够提供最大的 Q+U 的节点
        return: (action, node)的二元组
        """
        return max(self.children.items(), key=lambda action_node: action_node[1].value())

    def value(self) -> float:
        return (C_PUCT * self.P * np.sqrt(self.parent.n) / (1 + self.n)) + self.Q

    def update_node(self, leaf_value) -> None:
        """
        从叶节点评估中更新节点值
        leaf_value: 这个子节点的评估值来自当前玩家的视角
        """
        # 统计访问次数
        self.n += 1
        # 更新Q值，取决于所有访问次数的平均树，使用增量式更新方式
        self.Q += 1.0 * (leaf_value - self.Q) / self.n
        # self.update_value()
    
    def update_value(self) -> None:
        if self.parent is None:
            self._value = 0
        else:
            self._value = (C_PUCT * self.P * np.sqrt(self.parent.n) / (1 + self.n)) + self.Q
    

    # 使用递归的方法对所有节点（当前节点对应的支线）进行一次更新
    def update_all_node_in_path(self, leaf_value) -> None:
        """对所有直系父节点进行更新"""
        if self.parent is not None:
            self.parent.update_all_node_in_path(-leaf_value)
        self.update_node(leaf_value)

    def is_leaf(self) -> bool:
        """检查是否是叶节点，即没有被扩展的节点"""
        return len(self.children.items()) == 0

    def is_root(self) -> bool:
        return self.parent is None
