import numpy as np
import sys
sys.path.append("..")
from engine.action import Action
from engine.board import Board


def flip_action(action: Action) -> Action:
    from_x, from_y = Board.pos2xy(action.piece_from)
    to_x, to_y = Board.pos2xy(action.piece_to)
    from_x = 8 - from_x
    to_x = 8 - to_x
    from_pos = Board.xy2pos(from_x, from_y)
    to_pos = Board.xy2pos(to_x, to_y)
    return Action(from_pos, to_pos)


def flip_state(state: np.ndarray) -> np.ndarray:
    state_flip = state.transpose([1, 2, 0])
    state = state.transpose([1, 2, 0])
    for y in range(10):
        for x in range(9):
            state_flip[x][y] = state[8 - x][y]
    state_flip = state_flip.transpose([2, 0, 1])
    return state_flip
