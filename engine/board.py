import copy
import sys
sys.path.append("..")
import numpy as np
from engine.constant import Constant
from engine.piece import Force
from engine.piece import Piece
from engine.piece import PieceEnum
from engine.piece import PieceType
from engine.piece import CannonMoveStatus
from engine.piece import piece_one_hot


# Chinese Chess Board has 90 position.
class Board:
    # Initialize a 90 pos board
    def __init__(self, board: list = None):
        if board is None:
            self.board = [PieceEnum.EMPTY] * 90
            self.board[0] = PieceEnum.RED_ROOK
            self.board[1] = PieceEnum.RED_KNIGHT
            self.board[2] = PieceEnum.RED_BISHOP
            self.board[3] = PieceEnum.RED_GUARD
            self.board[4] = PieceEnum.RED_KING
            self.board[5] = PieceEnum.RED_GUARD
            self.board[6] = PieceEnum.RED_BISHOP
            self.board[7] = PieceEnum.RED_KNIGHT
            self.board[8] = PieceEnum.RED_ROOK
            self.board[19] = PieceEnum.RED_CANNON
            self.board[25] = PieceEnum.RED_CANNON
            self.board[27] = PieceEnum.RED_PAWN
            self.board[29] = PieceEnum.RED_PAWN
            self.board[31] = PieceEnum.RED_PAWN
            self.board[33] = PieceEnum.RED_PAWN
            self.board[35] = PieceEnum.RED_PAWN
            self.board[89] = PieceEnum.BLACK_ROOK
            self.board[88] = PieceEnum.BLACK_KNIGHT
            self.board[87] = PieceEnum.BLACK_BISHOP
            self.board[86] = PieceEnum.BLACK_GUARD
            self.board[85] = PieceEnum.BLACK_KING
            self.board[84] = PieceEnum.BLACK_GUARD
            self.board[83] = PieceEnum.BLACK_BISHOP
            self.board[82] = PieceEnum.BLACK_KNIGHT
            self.board[81] = PieceEnum.BLACK_ROOK
            self.board[70] = PieceEnum.BLACK_CANNON
            self.board[64] = PieceEnum.BLACK_CANNON
            self.board[62] = PieceEnum.BLACK_PAWN
            self.board[60] = PieceEnum.BLACK_PAWN
            self.board[58] = PieceEnum.BLACK_PAWN
            self.board[56] = PieceEnum.BLACK_PAWN
            self.board[54] = PieceEnum.BLACK_PAWN
        else:
            self.board = copy.deepcopy(board)

    def __getitem__(self, pos):
        return self.board[pos]

    def __setitem__(self, pos, item):
        self.board[pos] = item

    def available_pos(self, pos, force):
        return self.get_force(pos) != force

    def empty_pos(self, pos):
        return self.get_force(pos) == Force.EMPTY

    def get_force(self, pos):
        if pos == -1:
            raise Exception("Illegal pos")
        elif self.board[pos] == 0:
            return Force.EMPTY
        elif PieceEnum.RED_KING <= self.board[pos] <= PieceEnum.RED_PAWN:
            return Force.RED
        elif PieceEnum.BLACK_KING <= self.board[pos] <= PieceEnum.BLACK_PAWN:
            return Force.BLACK
        else:
            return Force.EMPTY

    def get_king_pos_by_force(self, current_force) -> int:
        for pos in range(Constant.TOTAL_POS):
            force, piece_type = Piece.piece_enum_2_piece_type_force(self.board[pos])
            if current_force == force and piece_type == PieceType.KING:
                return pos
        self.show()
        raise Exception("Failed to find king of force: ", current_force)

    def show(self, flag=False):
        if flag:
            print(self.board)
        board = [["" for _ in range(10)] for _ in range(11)]
        for i in range(90):
            name = Piece.get_name_by_enum(self.board[i])
            x = i % 9
            yt = i // 9
            y = 9 - yt
            # print(i, x, y)
            board[y][x + 1] = name
            board[y][0] = str(yt)
        for i in range(1, 10):
            board[10][i] = str(i - 1)
        board[10][0] = ""
        for y in range(0, 10, 1):
            for x in range(0, 10, 1):
                print(" %s " % board[y][x], end="")
            print("\n", end="")
        for x in range(10):
            print("%s   " % board[10][x], end="")
        print("\n", end="")

    @staticmethod
    def xy2pos(x: int, y: int) -> int:
        if Board.in_board(x, y):
            return y * Constant.NINE + x
        else:
            return -1

    @staticmethod
    def pos2xy(pos: int) -> (int, int):
        if pos == -1:
            return -1, -1
        else:
            return pos % Constant.NINE, pos // Constant.NINE

    @staticmethod
    def in_board(x: int, y: int) -> bool:
        return 0 <= x < Constant.NINE and 0 <= y < Constant.TEN

    @staticmethod
    def cal_pos_by_inc(pos, incx, incy: int) -> int:
        nx, ny = Board.pos2xy(pos)
        nx = nx + incx
        ny = ny + incy
        if Board.in_board(nx, ny):
            return Board.xy2pos(nx, ny)
        else:
            return -1

    @staticmethod
    def print(pos, new_pos: int, permit: bool, status: CannonMoveStatus) -> None:
        x, y = Board.pos2xy(pos)
        nx, ny = Board.pos2xy(new_pos)
        print(
            "(%d, %d) -> (%d, %d): permit: %s, status: %s" % (x, y, nx, ny, "Yes" if permit is True else "No", status))

    def calculate_total_exist_actions(self) -> None:
        cnt = 1530  # 车炮兵将(直线)
        knight_cnt = 0
        for pos in range(Constant.TOTAL_POS):
            x, y = Board.pos2xy(pos)
            if board.in_board(x + 1, y + 2):
                knight_cnt = knight_cnt + 1
            if board.in_board(x - 1, y + 2):
                knight_cnt = knight_cnt + 1
            if board.in_board(x + 1, y - 2):
                knight_cnt = knight_cnt + 1
            if board.in_board(x - 1, y - 2):
                knight_cnt = knight_cnt + 1
            if board.in_board(x + 2, y + 1):
                knight_cnt = knight_cnt + 1
            if board.in_board(x - 2, y + 1):
                knight_cnt = knight_cnt + 1
            if board.in_board(x + 2, y - 1):
                knight_cnt = knight_cnt + 1
            if board.in_board(x - 2, y - 1):
                knight_cnt = knight_cnt + 1
        bishop_advisor_cnt = 8 * 2 * 2 + 4 * 2 * 2
        print(cnt, knight_cnt, bishop_advisor_cnt, cnt + knight_cnt + bishop_advisor_cnt)

    def board_2_one_hot_tensor(self) -> np.ndarray:
        return board_2_one_hot_tensor(self.board)


def board_2_one_hot_tensor(param_board: []) -> np.ndarray:
    nda = np.zeros([9, 10, 7])
    for pos in range(Constant.TOTAL_POS):
        x, y = Board.pos2xy(pos)
        nda[x][y] = piece_one_hot[PieceEnum.get_name_by_value(param_board[pos])]
    return nda


# mirror board transform
def flip_board(ori_board: []) -> []:
    res = copy.deepcopy(ori_board)
    for y in range(Constant.TEN):
        for x in range(Constant.FOUR):
            nx = Constant.EIGHT - x
            pos1 = Board.xy2pos(x, y)
            pos2 = Board.xy2pos(nx, y)
            res[pos1], res[pos2] = res[pos2], res[pos1]
    return res


if __name__ == "__main__":
    arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 11, 0, 0, 0, 15, 0, 0, 0, 0, 0, 23, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0,
            27, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 14, 0, 0, 0, 21, 0, 0, 0]
    board = Board(arr)
    board.show()
    # board.calculate_total_exist_actions()
