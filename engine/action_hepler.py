import numpy as np
import sys
sys.path.append("..")
from engine.board import Board
from engine.action import Action
from engine.action import knight_step
from engine.constant import Constant


"""
Action转化，用于弥合算法和引擎之间对于数据定义的差异
"""
def generate_all_actions() -> (map, list):
    idx = 0
    action_2_id = {}
    id_2_action = [0 for _ in range(2086)]
    # straight-line action
    for pos in range(Constant.TOTAL_POS):
        x, y = Board.pos2xy(pos)
        for i in range(Constant.TEN):
            if i == y:
                continue
            action = Action(pos, Board.xy2pos(x, i))
            action_2_id[hash(action)] = idx
            id_2_action[idx] = action
            idx = idx + 1
        for i in range(Constant.NINE):
            if i == x:
                continue
            action = Action(pos, Board.xy2pos(i, y))
            action_2_id[hash(action)] = idx
            id_2_action[idx] = action
            idx = idx + 1
    # knight-action
    for pos in range(Constant.TOTAL_POS):
        x, y = Board.pos2xy(pos)
        for step in knight_step:
            nx = x + step[2]
            ny = y + step[3]
            if Board.in_board(nx, ny):
                action = Action(pos, Board.xy2pos(nx, ny))
                action_2_id[hash(action)] = idx
                id_2_action[idx] = action
                idx = idx + 1
    # bishop-action
    bishop_action_list = [(2, 18), (18, 2), (2, 22), (22, 2), (6, 22), (22, 6), (6, 26), (26, 6),
                          (38, 18), (18, 38), (38, 22), (22, 38), (42, 22), (22, 42), (42, 26), (26, 42),
                          (51, 71), (71, 51), (51, 67), (67, 51), (47, 67), (67, 47), (47, 63), (63, 47),
                          (87, 71), (71, 87), (87, 67), (67, 87), (83, 67), (67, 83), (83, 63), (63, 83)]
    for action_tuple in bishop_action_list:
        action = Action(*action_tuple)  # * is to divide the tuple.
        action_2_id[hash(action)] = idx
        id_2_action[idx] = action
        idx = idx + 1

    # advisor-action
    advisor_action_list = [(3, 13), (13, 3), (5, 13), (13, 5),
                           (21, 13), (13, 21), (23, 13), (13, 23),
                           (86, 76), (76, 86), (84, 76), (76, 84),
                           (68, 76), (76, 68), (66, 76), (76, 66)]
    for action_tuple in advisor_action_list:
        action = Action(*action_tuple)  # * is to divide the tuple.
        action_2_id[hash(action)] = idx
        id_2_action[idx] = action
        idx = idx + 1
    return action_2_id, id_2_action


action_2_id, id_2_action = generate_all_actions()

if __name__ == '__main__':
    a2i, i2a = action_2_id, id_2_action
    print("total actions: ", len(i2a))
