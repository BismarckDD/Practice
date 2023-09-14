from engine.observation import Observation
from engine.action_hepler import id_2_action, action_2_id
from engine.action import available_actions_with_check
from engine.action import Action
from engine.piece import Force
from engine.board import Board
import numpy as np
import copy
from collections import deque
import sys
sys.path.append("..")


class Game:

    def __init__(self):
        self.round = 1
        self.board = Board()
        self._is_terminated = False
        self.reward = [0.0, 0.0]
        self.history = deque()  # 用于保存历史board记录
        self.history.append(copy.deepcopy(self.board.board))
        self.action_2_id, self.id_2_action = action_2_id, id_2_action
        self.last_action_id: int = -1
        self.winner: int = -1
        self.kill_round: int = 200

    def reset(self) -> ():
        self.__init__()
        return self.get_observation()

    def get_current_force(self) -> Force:
        if self.round % 2 == 1:
            return Force.RED
        else:
            return Force.BLACK

    def get_round(self) -> int:
        return self.round

    def get_observation(self) -> Observation:
        available_actions = self.get_available_actions()
        if len(available_actions) == 0:
            self.terminate()
        if self.round > self.kill_round:
            self.tie()
        return Observation(available_actions, self.round, self._is_terminated, self.board.board, self.reward,
                           self.id_2_action[self.last_action_id], self.winner)

    def step(self, param_action: Action) -> Observation:
        self.round = self.round + 1
        self.board.board[param_action.piece_to] = self.board.board[param_action.piece_from]
        self.board.board[param_action.piece_from] = Force.EMPTY
        self.history.append(copy.deepcopy(self.board.board))
        self.last_action_id = self.action_2_id[hash(param_action)]
        return self.get_observation()

    # 1. get all actions of all pieces of current player.
    # 2. check.
    def get_available_actions(self) -> []:
        current_force = self.get_current_force()
        return available_actions_with_check(current_force, self.board)

    def terminate(self):
        self._is_terminated = True
        if self.get_current_force() == Force.RED:
            print("BLACK is the winner.")
            self.reward = [-1.0, 1.0]
            self.winner = Force.BLACK
        else:
            print("RED is the winner.")
            self.reward = [1.0, -1.0]
            self.winner = Force.RED
        self.show()

    def is_terminated(self) -> (bool, int):
        if self._is_terminated:
            return True, self.winner
        else:
            return False, None

    def tie(self):
        self._is_terminated = True
        print("Game is tie")
        self.reward = [0.0, 0.0]
        self.winner = -1
        self.show()

    def show(self, flag=True):
        print("当前回合: ", self.round)
        self.board.show(flag=flag)

        # 从当前玩家的视角返回棋盘状态，current_state_array: [9, 10, 9]  CHW

    def current_state(self):
        current_state = np.zeros([9, 9, 10])
        # 使用9个平面来表示棋盘状态
        # 0-6个平面表示棋子位置，1代表红方棋子，-1代表黑方棋子, 队列最后一个盘面
        # 第7个平面表示对手player最近一步的落子位置，走子之前的位置为-1，走子之后的位置为1，其余全部是0
        # 第8个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        # trans: 输出tensor的idx维是输入tensor的[idx]维
        # Tensor(9, 10, 7) -> Transform(2, 0, 1) -> Tensor(7, 9, 10)
        # 平面 0-6, board_2_state_tensor -> np.ndarray(9, 10, 7)
        current_state[:7] = self.board.board_2_one_hot_tensor().transpose([
            2, 0, 1])
        # 平面7, last_action
        if self.last_action_id != -1:
            action: Action = self.id_2_action[self.last_action_id]
            x1, y1 = Board.pos2xy(action.piece_from)
            x2, y2 = Board.pos2xy(action.piece_to)
            current_state[7][x1][y1] = -1
            current_state[7][x2][y2] = 1
        # 平面8, 先手玩家全1，后手玩家全0
        current_state[8][:, :] = (self.round % 2)
        return current_state


def show(tensor: np.ndarray) -> None:
    for i in range(9):
        print(i, tensor[i])


if __name__ == '__main__':
    game = Game()
    obs = game.reset()
    print(obs)
    show(game.current_state())
