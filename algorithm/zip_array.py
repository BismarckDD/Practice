import numpy as np
import copy
import time
from typing import Dict, Tuple
from config import CONFIG
from collections import deque  # 这个队列用来判断长将或长捉
import random
from engine.piece import PieceEnum

piece2array = dict({PieceEnum.RED_KING: np.array([1, 0, 0, 0, 0, 0, 0]),
                    PieceEnum.RED_ROOK: np.array([0, 1, 0, 0, 0, 0, 0]),
                    PieceEnum.RED_KNIGHT: np.array([0, 0, 1, 0, 0, 0, 0]),
                    PieceEnum.RED_CANNON: np.array([0, 0, 0, 1, 0, 0, 0]),
                    PieceEnum.RED_BISHOP: np.array([0, 0, 0, 0, 1, 0, 0]),
                    PieceEnum.RED_GUARD: np.array([0, 0, 0, 0, 0, 1, 0]),
                    PieceEnum.RED_PAWN: np.array([0, 0, 0, 0, 0, 0, 1]),
                    PieceEnum.BLACK_KING: np.array([-1, 0, 0, 0, 0, 0, 0]),
                    PieceEnum.BLACK_ROOK: np.array([0, -1, 0, 0, 0, 0, 0]),
                    PieceEnum.BLACK_KNIGHT: np.array([0, 0, -1, 0, 0, 0, 0]),
                    PieceEnum.BLACK_CANNON: np.array([0, 0, 0, -1, 0, 0, 0]),
                    PieceEnum.BLACK_BISHOP: np.array([0, 0, 0, 0, -1, 0, 0]),
                    PieceEnum.BLACK_GUARD: np.array([0, 0, 0, 0, 0, -1, 0]),
                    PieceEnum.BLACK_PAWN: np.array([0, 0, 0, 0, 0, 0, -1]),
                    PieceEnum.EMPTY: np.array([0, 0, 0, 0, 0, 0, 0])})


def array2piece(array):
    return list(filter(lambda string: (piece2array[string] == array).all(), piece2array))[0]


# 压缩存储
def state2OneHotState(state):
    tensor = np.zeros([10, 9, 7])
    for i in range(10):
        for j in range(9):
            tensor[i][j] = piece2array[state[i][j]]
    return tensor


# (state, mcts_prob, winner) ((9,10,9), 2086, 1) => ((9,90), (2,1043), 1)
def zip_state_mcts_prob(state: np.ndarray, action_prob: Dict[int, float], winner: float)\
    -> Tuple[np.ndarray, Dict[int, float], float]:
    state = state.reshape((9, -1))
    action_prob = action_prob.reshape((2, -1))
    state = zip_array(state)
    action_prob = zip_array(action_prob)
    return state, action_prob, winner


def recovery_state_mcts_prob(param: tuple)\
    -> Tuple[np.ndarray, Dict[int, float], float]:
    state, action_prob, winner = param
    state = recovery_array(state)
    action_prob = recovery_array(action_prob)
    state = state.reshape((9, 10, 9))
    action_prob = action_prob.reshape(2086)
    return state, action_prob, winner


# 注意，只能压缩二维数组
def zip_array(array: np.ndarray, data=0.) -> np.ndarray:  # 按照稀疏数组规范进行压缩
    zip_res = []
    # dim=2 is useless, 只是为了维度一致
    zip_res.append([len(array), len(array[0]), 0])
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array[i][j] != data:
                zip_res.append([i, j, array[i][j]])
    return np.array(zip_res)


def recovery_array(array: np.ndarray, data=0.) -> np.ndarray:  # 解压缩过程
    recovery_res = []
    for i in range(int(array[0][0])):
        recovery_res.append([data for i in range(int(array[0][1]))])
    for i in range(1, len(array)):
        recovery_res[int(array[i][0])][int(array[i][1])] = array[i][2]
    return np.array(recovery_res)
