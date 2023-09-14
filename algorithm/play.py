import sys
sys.path.append("..")
import time
import numpy as np
from typing import Dict, List, Tuple
from engine.action import Action
from engine.game import Game
from agent.mcts_agent import MctsAgent
from algorithm.policy_value_net import PolicyValueNet


def start_self_play(agent: MctsAgent) -> (int, List[Tuple[np.ndarray, Dict[int, float], float]]):
    game = Game()
    state_list, action_probs_list, current_players = [], [], []
    # 开始自我对弈
    while True:
        start_time = time.time()
        game.show(False)
        action, action_probs = agent.get_action_in_training(
            game.get_observation())
        # print("Game Round: %d, Step Cost: %d" % (game.get_round(), time.time() - start_time))
        # print(action)
        # 保存自我对弈的数据
        state_list.append(game.current_state())
        action_probs_list.append(action_probs)
        current_players.append(game.get_current_force())
        # 执行一步落子
        game.step(action)
        print("execute: ", action)
        #
        is_terminated, winner = game.is_terminated()
        if is_terminated:
            # 从每一个状态state对应的玩家的视角保存胜负信息
            winner_z = np.zeros(len(current_players))
            if winner != -1:
                winner_z[np.array(current_players) == winner] = 1.0
                winner_z[np.array(current_players) != winner] = -1.0
            # 重置蒙特卡洛根节点
            agent.reset()
            if winner != -1:
                print("Game is Finished. Winner is:", winner)
            else:
                print("Game is Finished. Tie")
            # play_data = [(state, action_prob, value)]
            play_data = list(zip(state_list, action_probs_list, winner_z))
            # print(type(play_data))
            # print(type(play_data[0][0]))
            # print(type(play_data[0][1]))
            # print(type(play_data[0][2]))
            return winner, play_data


if __name__ == "__main__":
    policy_value_net = PolicyValueNet(checkpoint_file="current_model.pkl", type="train")
    mcts_agent = MctsAgent(
        policy_value_function=policy_value_net.policy_value_fn,
        n_playout=10)

    a, b = start_self_play(mcts_agent)
    print("Self play is finished. res %d" % a)
    # for item in list(b):
    #     print(item[0], item[1], item[2])
