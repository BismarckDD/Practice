from engine.observation import Observation
from engine.action import Action
from engine.board import Board
from agent.base_agent import BaseAgent
from engine.piece import Piece


class HumanAgent(BaseAgent):

    def __init__(self):
        super().__init__()

    def step(self, obs: Observation) -> Action:
        if obs.round % 2 == 1:
            aspect = "红棋"
        else:
            aspect = "黑棋"
        # print("当前回合：", obs.round, "您执", aspect)
        # print("当前棋盘状态: ")
        # board = Board(obs.board)
        # board.show()
        print("请选择指令(输入序号)：")
        actions = obs.available_actions
        actions = sorted(actions)
        while True:

            try:
                cnt = 1
                for action in actions:
                    piece_from = Board.pos2xy(action.piece_from)
                    piece_to = Board.pos2xy(action.piece_to)
                    print("%d. (%d, %d) -> (%d, %d), type: %s" %
                          (cnt, piece_from[0], piece_from[1], piece_to[0], piece_to[1],
                           Piece.get_name_by_type(action.piece_type)))
                    cnt = cnt + 1
                i = input()
                i = int(i) - 1
                if 0 <= i < len(actions):
                    return actions[i]
                else:
                    print("请重新输入选项")
            except Exception as e:
                print("请重新输入")
            except KeyboardInterrupt as ki:
                exit(0)
