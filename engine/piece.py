# coding=utf-8

from enum import Enum
import numpy as np
import sys

sys.path.append("..")
from engine.constant import Constant


class Force(object):
    EMPTY = 0
    RED = 1
    BLACK = 2
    UNKNOWN = 3


class PieceType(object):
    ILLEGAL = 0
    KING = 1
    ROOK = 2
    KNIGHT = 3
    CANNON = 4
    BISHOP = 5
    GUARD = 6
    PAWN = 7
    UNKNOWN = 8


class PieceEnum(object):
    EMPTY = 0
    RED_KING = 11
    RED_ROOK = 12
    RED_KNIGHT = 13
    RED_CANNON = 14
    RED_BISHOP = 15
    RED_GUARD = 16
    RED_PAWN = 17
    BLACK_KING = 21
    BLACK_ROOK = 22
    BLACK_KNIGHT = 23
    BLACK_CANNON = 24
    BLACK_BISHOP = 25
    BLACK_GUARD = 26
    BLACK_PAWN = 27

    @staticmethod
    def get_name_by_value(value: int):
        if value == 0:
            return 'EMPTY'
        elif value == 11:
            return 'RED_KING'
        elif value == 12:
            return 'RED_ROOK'
        elif value == 13:
            return 'RED_KNIGHT'
        elif value == 14:
            return 'RED_CANNON'
        elif value == 15:
            return 'RED_BISHOP'
        elif value == 16:
            return 'RED_GUARD'
        elif value == 17:
            return 'RED_PAWN'
        elif value == 21:
            return 'BLACK_KING'
        elif value == 22:
            return 'BLACK_ROOK'
        elif value == 23:
            return 'BLACK_KNIGHT'
        elif value == 24:
            return 'BLACK_CANNON'
        elif value == 25:
            return 'BLACK_BISHOP'
        elif value == 26:
            return 'BLACK_GUARD'
        elif value == 27:
            return 'BLACK_PAWN'
        else:
            return 'ILLEGAL'


class CannonMoveStatus(Enum):
    DIRECT = 0
    EMMIT = 1
    FINNISH = 2


# 这里用-1表示黑子，是否合适？ 经过Doctor JianingLi确认，这里没有问题
piece_one_hot = dict(RED_KING=np.array([1, 0, 0, 0, 0, 0, 0]),
                     RED_ROOK=np.array([0, 1, 0, 0, 0, 0, 0]),
                     RED_KNIGHT=np.array([0, 0, 1, 0, 0, 0, 0]),
                     RED_CANNON=np.array([0, 0, 0, 1, 0, 0, 0]),
                     RED_BISHOP=np.array([0, 0, 0, 0, 1, 0, 0]),
                     RED_GUARD=np.array([0, 0, 0, 0, 0, 1, 0]),
                     RED_PAWN=np.array([0, 0, 0, 0, 0, 0, 1]),
                     BLACK_KING=np.array([-1, 0, 0, 0, 0, 0, 0]),
                     BLACK_ROOK=np.array([0, -1, 0, 0, 0, 0, 0]),
                     BLACK_KNIGHT=np.array([0, 0, -1, 0, 0, 0, 0]),
                     BLACK_CANNON=np.array([0, 0, 0, -1, 0, 0, 0]),
                     BLACK_BISHOP=np.array([0, 0, 0, 0, -1, 0, 0]),
                     BLACK_GUARD=np.array([0, 0, 0, 0, 0, -1, 0]),
                     BLACK_PAWN=np.array([0, 0, 0, 0, 0, 0, -1]),
                     EMPTY=np.array([0, 0, 0, 0, 0, 0, 0]))


class Piece:
    # force: 棋子所属阵营: 红or黑
    # type: 棋子类型: 帅车马炮相士
    # piece_enum = force + type
    def __init__(self, piece_enum, pos):
        self.piece_enum = piece_enum
        self.pos = pos

    @staticmethod
    def get_name_by_enum(e):
        if e == PieceEnum.RED_KING:
            return "帅"
        elif e == PieceEnum.RED_ROOK:
            return "俥"
        elif e == PieceEnum.RED_KNIGHT:
            return "傌"
        elif e == PieceEnum.RED_CANNON:
            return "炮"
        elif e == PieceEnum.RED_BISHOP:
            return "相"
        elif e == PieceEnum.RED_GUARD:
            return "仕"
        elif e == PieceEnum.RED_PAWN:
            return "兵"
        elif e == PieceEnum.BLACK_KING:
            return "将"
        elif e == PieceEnum.BLACK_ROOK:
            return "車"
        elif e == PieceEnum.BLACK_KNIGHT:
            return "马"
        elif e == PieceEnum.BLACK_CANNON:
            return "砲"
        elif e == PieceEnum.BLACK_BISHOP:
            return "象"
        elif e == PieceEnum.BLACK_GUARD:
            return "士"
        elif e == PieceEnum.BLACK_PAWN:
            return "卒"
        else:
            return "＋"

    @staticmethod
    def get_name_by_type(piece_type):
        if piece_type == PieceType.KING:
            return "帅"
        elif piece_type == PieceType.ROOK:
            return "俥"
        elif piece_type == PieceType.KNIGHT:
            return "傌"
        elif piece_type == PieceType.CANNON:
            return "炮"
        elif piece_type == PieceType.BISHOP:
            return "相"
        elif piece_type == PieceType.GUARD:
            return "仕"
        elif piece_type == PieceType.PAWN:
            return "兵"
        else:
            return "*"

    @staticmethod
    def piece_enum_2_piece_type_force(piece_enum: PieceEnum) -> (Force, PieceType):
        return piece_enum // Constant.TEN, piece_enum % Constant.TEN


if __name__ == "__main__":
    print(piece_one_hot)
    # print(PieceEnum.__members__.values())
    # for item in PieceEnum.__members__.values():
    #     print(item.name, " ", item.value)
    #     if item.value == PieceEnum.RED_ROOK:
    #         print("Found")

    if PieceEnum.EMPTY == Force.EMPTY:
        print("Equal")
    else:
        print("Error")
