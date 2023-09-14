import sys
sys.path.append("..")
import pickle
import time
import random
import numpy as np
import zip_array
from collections import defaultdict
from collections import deque
from engine.game import Game
from policy_value_net import PolicyValueNet
from config import CONFIG
from agent.mcts_agent import MctsAgent
from agent.pure_mcts_agent import PureMctsAgent
from algorithm.config import TRAIN_DATA_BUFFER_PATH, DATA_BUFFER, ITERS, MODEL_PATH

train_data_file_path = CONFIG[TRAIN_DATA_BUFFER_PATH]
checkpoint_file_path = CONFIG[MODEL_PATH]

class TrainPipeline:
    def __init__(self, checkpoint_file=None):
        # 训练参数
        self.game = Game()
        self.n_playout = CONFIG["play_out"]
        self.c_puct = CONFIG["c_puct"]
        self.learn_rate = 1e-3
        self.lr_multiplier = 1  # 基于KL自适应的调整学习率
        self.temp = CONFIG["temperature"]
        self.batch_size = CONFIG["batch_size"]  # 训练的batch大小
        self.epochs = CONFIG["epoch"]  # 每次更新, train_step的数值
        self.kl_targ = CONFIG["kl_targ"]  # kl散度控制
        self.check_freq = CONFIG["check_freq"]  # 保存模型的频率
        self.game_batch_num = CONFIG["game_batch_num"]  # 训练更新的次数
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 500                #
        self.buffer_size = CONFIG["buffer_size"]
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.checkpoint_file = checkpoint_file
        self.init_policy_value_net()
    
    def init_policy_value_net(self):
        if self.checkpoint_file is not None:
            try:
                self.policy_value_net = PolicyValueNet(checkpoint_file=self.checkpoint_file, type="train")
                print("Load ckpt succeed.")
            except Exception as e:
                # 从零开始训练
                print("Load ckpt failed, train from scratch.")
                self.policy_value_net = PolicyValueNet(checkpoint_file=None, type="train")
        else:
            print("Train from scratch")
            self.policy_value_net = PolicyValueNet(checkpoint_file=None, type="train")

    # 感觉并没什么卵用
    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        mcts_agent = MctsAgent(self.policy_value_net.policy_value_fn,
                               c_puct=self.c_puct,
                               n_playout=self.n_playout)
        pure_mcts_agent = PureMctsAgent(c_puct=5,
                                        n_playout=self.pure_mcts_playout_num)
        # what's the win_cnt structure?
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(mcts_agent,
                                          pure_mcts_agent,
                                          start_player=i % 2 + 1,
                                          is_shown=1)
            win_cnt[winner] += 1
        # win +1, tie +0.5
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    # 训练用fp32, 推理用fp16是不是比较好
    def policy_update(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        # print(mini_batch[0][1],mini_batch[1][1])
        mini_batch = [zip_array.recovery_state_mcts_prob(data) for data in mini_batch]

        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch)

        action_probs_batch = [data[1] for data in mini_batch]
        action_probs_batch = np.array(action_probs_batch)

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch)

        # 旧的策略，旧的价值函数
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        # 训练时所用的数据：棋盘状态, mcts输出的状态, 对局最终的结果, 以及一个学习率
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                action_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            # 新的策略，新的价值函数
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # 如果KL散度很差，则提前终止训练过程(这里做一次推理的目的只是为了计算散度，以提前终止训练过程)
                break

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # print(old_v.flatten(),new_v.flatten())
        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print(("kl:{:.5f}, learning_rate:{:.5f}, lr_multiplier:{:.3f}, "
               "loss:{}, entropy:{}, explained_var_old:{:.9f}, explained_var_new:{:.9f}"
               ).format(kl,
                        self.learn_rate,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def run(self):
        """开始训练"""
        cnt = 0
        while True:
            if cnt % self.check_freq == 0:
                try:
                    with open(train_data_file_path, "rb") as data_handle:
                        data_file = pickle.load(data_handle)
                        self.data_buffer = data_file[DATA_BUFFER]
                        self.iters = data_file[ITERS]
                        del data_file
                    print("Load train data succeed, execute policy_update.")
                except:
                    print("Load train data failed, sleep 30 seconds.")
                    time.sleep(30)
            cnt = cnt + 1
            try:
                if len(self.data_buffer) >= self.batch_size:
                    loss, entropy = self.policy_update()
                    self.policy_value_net.save_model(checkpoint_file_path)
                    print("Epoch %d is finished." % cnt)
                    if cnt % self.check_freq == 0:
                        print("Save current checkpoint file: {}, reload policy_value net.".format(cnt))
                        self.policy_value_net.save_model("models/current_policy_batch{}.model".format(cnt))
                else:
                    print("No enought data, sleep 30s.")
                    time.sleep(30)
                    cnt = cnt - 1
            except KeyboardInterrupt:
                print("\n\rTraining Process is Finished.")
                break


if __name__ == '__main__':
    train_pipeline = TrainPipeline(checkpoint_file='current_model.pkl')
    train_pipeline.run()
