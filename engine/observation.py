import numpy as np
import copy
import sys
sys.path.append("..")
from engine.action import Action
from engine.action import available_actions_with_check
from engine.action_hepler import action_2_id, id_2_action
from engine.board import board_2_one_hot_tensor
from engine.board import Board
from engine.piece import Force


class Observation:

    def __init__(self, available_actions: list = [], round: int = 1, is_terminated: bool = False,
                 board: [] = [], reward: [] = [0.0, 0.0], action: Action = None, winner: int = -1):
        self.round = round
        self._is_terminated = is_terminated
        self.available_actions = available_actions
        self.board = copy.deepcopy(board)
        self.reward = reward
        self.last_action:Action = action  # 上一招棋，Action
        self.winner = -1
        self.current_player = Force.RED if self.round % 2 == 1 else Force.BLACK

    def __str__(self):
        return "round: %s, is_terminate: %s, available_actions: %s, board: %s" % \
            (self.round, self._is_terminated, self.available_actions, self.board)

    def current_state(self):
        current_state = np.zeros([9, 9, 10])
        # 使用9个平面来表示棋盘状态
        # 0-6个平面表示棋子位置，1代表红方棋子，-1代表黑方棋子, 队列最后一个盘面
        # 第7个平面表示对手player最近一步的落子位置，走子之前的位置为-1，走子之后的位置为1，其余全部是0
        # 第8个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        # trans: 输出tensor的idx维是输入tensor的[idx]维
        # Tensor(9, 10, 7) -> Transform(2, 0, 1) -> Tensor(7, 9, 10)
        # 平面 0-6, board_2_state_tensor -> np.ndarray(9, 10, 7)
        current_state[:7] = board_2_one_hot_tensor(self.board).transpose([2, 0, 1])
        # 平面7, last_action
        if self.last_action is not None:
            x1, y1 = Board.pos2xy(self.last_action.piece_from)
            x2, y2 = Board.pos2xy(self.last_action.piece_to)
            current_state[7][x1][y1] = -1
            current_state[7][x2][y2] = 1
        # 平面8, 先手玩家全1，后手玩家全0
        current_state[8][:, :] = (self.round % 2)
        return current_state

    def is_terminated(self) -> (bool, int):
        return (False, None) if self._is_terminated is False else (True, self.winner)

    # 本方法用在mcts中，正常规范里是不需要的
    def step(self, action_id: int) -> None:
        action: Action = id_2_action[action_id]
        self.last_action = action
        self.board[action.piece_to] = self.board[action.piece_from]
        self.board[action.piece_from] = 0
        self.round = self.round + 1
    
    def get_status(self) -> None:
        current_force = Force.RED if self.round % 2 == 1 else Force.BLACK
        _board = Board(self.board)
        self.available_actions = available_actions_with_check(current_force, _board)
        if len(self.available_actions) == 0:
            self._is_terminated = True
            self.winner = Force.BLACK if self.round % 2 == 1 else Force.RED

    def current_player_id(self):
        self.current_player = Force.RED if self.round % 2 == 1 else Force.BLACK
        return self.current_player
    
    def show(self):
        print("round: %d" % self.round)
        print("is_terminate: %s" % self._is_terminated)
        print("last_action: %s" % str(self.last_action))
        print("board: ", self.board)
        print("available action length: %d" % len(self.available_actions))


