"""自我对弈收集数据"""
# coding=utf-8
import sys
sys.path.append("..")
from typing import List, Tuple, Dict
from algorithm.flip_helper import flip_action, flip_state
from algorithm.play import start_self_play
from algorithm.config import TRAIN_DATA_BUFFER_PATH, DATA_BUFFER, ITERS
from algorithm.policy_value_net import PolicyValueNet
from algorithm.zip_array import zip_state_mcts_prob
from config import CONFIG
from agent.pure_mcts_agent import PureMctsAgent
from agent.mcts_agent import MctsAgent
from engine.action_hepler import id_2_action, action_2_id
from engine.game import Game
import pickle
import os
import copy
from collections import deque
import numpy as np
import time
import multiprocessing

train_data_file_path = CONFIG[TRAIN_DATA_BUFFER_PATH]

# 定义整个对弈收集数据流程
class CollectPipeline:

    def __init__(self, checkpoint_file=None):
        self.game = Game()
        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = CONFIG["play_out"]  # 每次移动的模拟次数
        self.c_puct = CONFIG["c_puct"]  # u的权重
        self.buffer_size = CONFIG["buffer_size"]  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.checkpoint_file = checkpoint_file
        self.mcts_agent = None  # used for self-play to generate data.
        self.policy_value_net = None
        self.episode_len = 0
        self.lock = multiprocessing.Lock()

    # 从主体加载模型
    def init_policy_value_net(self):
        try:
            self.policy_value_net = PolicyValueNet(checkpoint_file=self.checkpoint_file)
            print("Use existed models.: %s" % self.checkpoint_file)
        except Exception as e:
            self.policy_value_net = PolicyValueNet()
            print("Use new models.")
        self.mcts_agent = MctsAgent(
            policy_value_function=self.policy_value_net.policy_value_fn,
            n_playout=self.n_playout)
        # self.pure_mcts_agent = PureMctsAgent()

    def add_mirror_data(self, play_data: List[Tuple[np.ndarray, Dict[int, float], float]])\
        -> List[Tuple[np.ndarray, Dict[int, float], float]]:
        """左右对称变换，扩充数据集一倍，加速一倍训练速度"""
        mirror_data = []
        # 棋盘状态[9, 9, 10], 走子概率，赢家
        # C=9, W=9, H=10:
        # transpose(1,2,0): (9,10,9) ->(10, 9, 9) dim[idx] = dim[trans[idx]]
        # transpose(2,0,1): (10,9,9) ->(9, 10, 9) dim[idx] = dim[trans[idx]]
        for (state, action_probs, winner) in play_data:
            # 原始数据
            mirror_data.append(zip_state_mcts_prob(state, action_probs, winner))
            # 水平翻转后的数据
            state_flip = flip_state(state)
            action_probs_flip = copy.deepcopy(action_probs)
            # action_probs is a ndarray (2086, )
            for action_id in range(action_probs.shape[0]):
                prob = action_probs[action_id]
                action_id_flip = action_2_id[hash(
                    flip_action(id_2_action[action_id]))]
                # mcts_prob_flip[new_action_id] = mcts_prob[action_id]
                action_probs_flip[action_id_flip] = prob
            mirror_data.append(zip_state_mcts_prob(
                state_flip, action_probs_flip, winner))
        return mirror_data

    def collect_self_play_data(self, lock: multiprocessing.Lock, n_games: int=1) -> int:
        # 收集自我对弈的数据
        for i in range(n_games):
            try:
                policy_value_net = PolicyValueNet(checkpoint_file=self.checkpoint_file)
                print("Use existed models.: %s" % self.checkpoint_file)
            except Exception as e:
                policy_value_net = PolicyValueNet()
                print("Use new models.")
            mcts_agent = MctsAgent(policy_value_function=policy_value_net.policy_value_fn,
                                   n_playout=self.n_playout)
            winner, play_data = start_self_play(mcts_agent)
            # play_data = list(play_data)[:]  # 单纯复制一份？
            self.episode_len = len(play_data)  # episode，进行了多少个episode
            # 获得镜像数据，对棋盘做镜像对称，数据仍然可以使用
            play_data = self.add_mirror_data(play_data)
            lock.acquire()
            try:
                if os.path.exists(train_data_file_path):
                    with open(train_data_file_path, "rb") as train_data_file:
                        data_dict = pickle.load(train_data_file)
                        self.data_buffer = deque(maxlen=self.buffer_size)
                        self.data_buffer.extend(data_dict[DATA_BUFFER])
                        self.iters = data_dict[ITERS]
                        del data_dict  # is necessary?
                self.data_buffer.extend(play_data)
                self.iters += 1
                data_dict = {DATA_BUFFER: self.data_buffer, ITERS: self.iters}
                with open(train_data_file_path, "wb") as train_data_file:
                    pickle.dump(data_dict, train_data_file)
                print("%d games finished." % self.iters)
            finally:
                lock.release()
        return self.iters

    def run(self):
        """开始收集数据"""
        process_num  = 1
        processes = []
        lock = multiprocessing.Lock()
        try:
            for i in range(process_num):
                p = multiprocessing.Process(target=self.collect_self_play_data, args=[lock, 10000])
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            # print("Batch: {}, Episode_Len: {}".format(iters, self.episode_len))
        except KeyboardInterrupt:
            print("\n\rquit")


if __name__ == "__main__":
    # 通过当前已有的model构建网络，model可以不存在
    collecting_pipeline = CollectPipeline(checkpoint_file="current_model.pkl")
    collecting_pipeline.run()
