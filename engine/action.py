import sys
sys.path.append("..")
from engine.constant import Constant
from engine.board import Board
from engine.piece import Piece
from engine.piece import CannonMoveStatus
from engine.piece import Force
from engine.piece import PieceType
import copy

# knight move step
king_step = [[-1, 0], [1, 0], [0, -1], [0, 1]]
red_king_pos = {}
black_king_pos = {}
# knight move step
knight_step = [[1, 0, 2, 1], [1, 0, 2, -1], [-1, 0, -2, 1], [-1, 0, -2, -1],
               [0, 1, 1, 2], [0, 1, -1, 2], [0, -1, -1, -2], [0, -1, 1, -2]]
# bishop move scope
bishop_step = [[1, 1, 2, 2], [1, -1, 2, -2], [-1, 1, -2, 2], [-1, -1, -2, -2]]
red_bishop_pos = {2, 6, 18, 22, 26, 38, 42}
black_bishop_pos = {89, 81, 71, 67, 63, 51, 47}
# guard move scope
guard_step = [[1, -1], [1, 1], [-1, 1], [-1, -1]]
red_guard_pos = {}
black_guard_pos = {}

RED_LIMIT = 44
BLACK_LIMIT = 45


class Action:

    def __init__(self):
        self.piece_from = 0
        self.piece_to = 0
        self.force = 0
        self.piece_type = 0

    def __init__(self, piece_from, piece_to: int, force: Force = Force.RED, piece_type: PieceType = PieceType.KING):
        self.piece_from = piece_from
        self.piece_to = piece_to
        self.force = force
        self.piece_type = piece_type

    def __lt__(self, other) -> bool:
        # 自定义比较逻辑，比较对象的value属性
        if self.piece_type != other.piece_type:
            return self.piece_type < other.piece_type
        elif self.force != other.force:
            return self.force < other.force
        elif self.piece_from != other.piece_from:
            return self.piece_from < other.piece_from
        else:
            return self.piece_to < other.piece_to

    def __hash__(self):
        return self.piece_from * 100 + self.piece_to

    def __str__(self):
        fx, fy = Board.pos2xy(self.piece_from)
        tx, ty = Board.pos2xy(self.piece_to)
        return "From: (%d, %d); To: (%d, %d)" % (fx, fy, tx, ty)


# 带将军检测的actions方法
def available_actions_with_check(current_force: Force, board: Board) -> []:
    # 1. 获取本方所有的可用action
    actions = available_actions(current_force, board)
    final_actions = []
    # 2. check这些action是否会造成将军
    for action in actions:
        if check_action(current_force, action, board) is True:
            final_actions.append(action)
    return final_actions


# 不带将军检测的actions方法
def available_actions(current_force: Force, board: Board) -> []:
    actions = []
    for pos in range(Constant.TOTAL_POS):
        force, piece_type = Piece.piece_enum_2_piece_type_force(board[pos])
        # print("pos: %d, force: %d, cforce: %s type: %s" % (pos, force, current_force, piece_type))
        if force is current_force:
            if piece_type is PieceType.KING:
                actions.extend(king_actions(current_force, pos, board))
            elif piece_type == PieceType.ROOK:
                actions.extend(rook_actions(current_force, pos, board))
            elif piece_type == PieceType.KNIGHT:
                actions.extend(knight_actions(current_force, pos, board))
            elif piece_type == PieceType.CANNON:
                # print(force, current_force)
                actions.extend(cannon_actions(current_force, pos, board))
            elif piece_type == PieceType.BISHOP:
                actions.extend(bishop_actions(current_force, pos, board))
            elif piece_type == PieceType.GUARD:
                actions.extend(guard_actions(current_force, pos, board))
            elif piece_type == PieceType.PAWN:
                actions.extend(pawn_actions(current_force, pos, board))
    return actions


# True: rivial check self-king
# False: rival doesn't check.
def cover_king(board: Board, src_pos, des_pos, piece_type: int) -> bool:
    if piece_type in (PieceType.BISHOP, PieceType.GUARD):
        return False
    sx, sy = Board.pos2xy(src_pos)
    dx, dy = Board.pos2xy(des_pos)
    if piece_type == PieceType.ROOK:
        if sx == dx:
            if sy > dy:
                sy, dy = dy, sy
            for i in range(sy + 1, dy):
                if board.empty_pos(Board.xy2pos(sx, i)) is False:
                    return False
            return True
        elif sy == dy:
            if sx > dx:
                sx, dx = dx, sx
            for i in range(sx + 1, dx):
                if board.empty_pos(Board.xy2pos(i, sy)) is False:
                    return False
            return True
        else:
            return False
    elif piece_type == PieceType.KNIGHT:
        mx = dx - sx
        my = dy - sy
        delta_x = 0
        delta_y = 0
        p = mx * my
        p = p if p >= 0 else -p
        if p == 2:
            if mx == 2:
                delta_x = 1
            elif mx == -2:
                delta_x = -1
            elif my == 2:
                delta_y = 1
            elif my == -2:
                delta_y = -1
            ex = sx + delta_x
            ey = sy + delta_y
            # print(ex, ey, sx, sy, dx, dy, delta_x, delta_y)
            return board.empty_pos(Board.xy2pos(ex, ey))
        else:
            return False
    elif piece_type == PieceType.CANNON:
        cnt = 0
        if sx == dx:
            if sy > dy:
                sy, dy = dy, sy
            for i in range(sy + 1, dy):
                if board.empty_pos(Board.xy2pos(sx, i)) is False:
                    cnt = cnt + 1
            return cnt == 1
        elif sy == dy:
            if sx > dx:
                sx, dx = dx, sx
            for i in range(sx + 1, dx):
                if board.empty_pos(Board.xy2pos(i, sy)) is False:
                    cnt = cnt + 1
            return cnt == 1
        else:
            return False
    elif piece_type == PieceType.PAWN:
        if sx == dx:
            if dy >= 6:
                return dy == sy + 1
            else:
                return dy == sy - 1
        elif sy == dy:
            return sx == dx + 1 or sx == dx - 1
        else:
            return False
    elif piece_type == PieceType.KING:
        if sx == dx:
            if sy > dy:
                sy, dy = dy, sy
            for i in range(sy + 1, dy, 1):
                if board.empty_pos(Board.xy2pos(sx, i)) is False:
                    return False
            return True
        else:
            return False
    else:
        return False

# 保证做出动作后己方不能处于被check状态
def check_action(force: Force, action: Action, board: Board) -> bool:
    # 检测执行过action之后是否会将军
    # Board构造函数内部会deep copy,这里不需要了
    new_board = Board(board.board)
    new_board[action.piece_to] = new_board[action.piece_from]
    new_board[action.piece_from] = Force.EMPTY
    # 本方king的位置
    king_pos = new_board.get_king_pos_by_force(force)
    rival_force = Force.BLACK if force == Force.RED else Force.RED
    for i in range(90):
        if new_board.get_force(i) == rival_force:
            piece_type = new_board[i] % Constant.TEN
            if cover_king(new_board, i, king_pos, piece_type):
                return False
    #rival_actions = available_actions(rival_force, new_board)
    #for rival_action in rival_actions:
    #    if rival_action.piece_to == king_pos:
    #        return False
    # rival_king_pos = new_board.get_king_pos_by_force(rival_force)
    # # 检查两个老将照脸的情况
    # x1, y1 = Board.pos2xy(king_pos)
    # x2, y2 = Board.pos2xy(rival_king_pos)
    # if x1 == x2:
    #     if y1 > y2:
    #         y1, y2 = y2, y1
    #     for y in range(y1 + 1, y2, 1):
    #         tmp_pos = Board.xy2pos(x1, y)
    #         # 2024.01.05 board.empty_pos -> new_board.empty_pos, solve a bug.
    #         if new_board.empty_pos(tmp_pos) is not True:
    #             return True
    #     return False
    return True


def in_9grid(force, pos):
    if force == Force.RED and pos in (3, 4, 5, 12, 13, 14, 21, 22, 23) or \
            force == Force.BLACK and pos in (84, 85, 86, 75, 76, 77, 66, 67, 68):
        return True
    else:
        return False


def decode_action(encode_data):
    piece_from = encode_data // 100
    piece_to = encode_data % 100
    return Action(piece_from, piece_to, Force.UNKNOWN, PieceType.UNKNOWN)


def encode_action(actoin: Action) -> int:
    return actoin.piece_from * 100 + piece_to


# 王的action list
def king_actions(force: Force, pos: int, board: Board) -> []:
    actions = []
    new_pos = Board.cal_pos_by_inc(pos, 1, 0)
    if in_9grid(force, new_pos) and board.available_pos(new_pos, force):
        actions.append(Action(pos, new_pos, force, PieceType.KING))
    new_pos = Board.cal_pos_by_inc(pos, -1, 0)
    if in_9grid(force, new_pos) and board.available_pos(new_pos, force):
        actions.append(Action(pos, new_pos, force, PieceType.KING))
    new_pos = Board.cal_pos_by_inc(pos, 0, 1)
    if in_9grid(force, new_pos) and board.available_pos(new_pos, force):
        actions.append(Action(pos, new_pos, force, PieceType.KING))
    new_pos = Board.cal_pos_by_inc(pos, 0, -1)
    if in_9grid(force, new_pos) and board.available_pos(new_pos, force):
        actions.append(Action(pos, new_pos, force, PieceType.KING))
    return actions


#
def rook_actions_helper(force: Force, pos: int, new_pos: int, board: Board) -> ([], bool):
    if new_pos != -1:  # 棋盘内区域
        # if board.get_force(new_pos) != force:  # 空区域或对方棋子
        if board.get_force(new_pos) == Force.EMPTY:
            return Action(pos, new_pos, force, PieceType.ROOK), False
        if board.get_force(new_pos) != force:
            return Action(pos, new_pos, force, PieceType.ROOK), True
        else:  # 遇到己方棋子
            return None, True
    else:  # 棋盘外区域
        return None, True


# 车的action list
def rook_actions(force: Force, pos: int, board: Board) -> []:
    actions = []
    # -> right
    for i in range(1, 9, 1):
        new_pos = board.cal_pos_by_inc(pos, i, 0)
        action, terminate = rook_actions_helper(force, pos, new_pos, board)
        # Board.print(pos, new_pos, action is not None, 0)
        if action is not None:
            actions.append(action)
        if terminate is True:
            break
    # -> left
    for i in range(-1, -9, -1):
        new_pos = board.cal_pos_by_inc(pos, i, 0)
        action, terminate = rook_actions_helper(force, pos, new_pos, board)
        # Board.print(pos, new_pos, action is not None, 0)
        if action is not None:
            actions.append(action)
        if terminate is True:
            break
    # -> up
    for i in range(1, 10, 1):
        new_pos = board.cal_pos_by_inc(pos, 0, i)
        action, terminate = rook_actions_helper(force, pos, new_pos, board)
        # Board.print(pos, new_pos, action is not None, 0)
        if action is not None:
            actions.append(action)
        if terminate is True:
            break
    # -> down
    for i in range(-1, -10, -1):
        new_pos = board.cal_pos_by_inc(pos, 0, i)
        action, terminate = rook_actions_helper(force, pos, new_pos, board)
        # Board.print(pos, new_pos, action is not None, 0)
        if action is not None:
            actions.append(action)
        if terminate is True:
            break
    return actions


def knight_actions(force: Force, pos: int, board: Board) -> []:
    actions = []
    ori_x, ori_y = Board.pos2xy(pos)
    for item in knight_step:
        obstacle = (ori_x + item[0], ori_y + item[1])
        if board.in_board(obstacle[0], obstacle[1]) is False:
            continue
        obstacle_pos = board.xy2pos(obstacle[0], obstacle[1])
        if board.empty_pos(obstacle_pos) is not True:
            continue
        nx, ny = (ori_x + item[2], ori_y + item[3])
        if board.in_board(nx, ny) is False:
            continue
        next_p = board.xy2pos(nx, ny)
        if board.available_pos(next_p, force):
            actions.append(Action(pos, next_p, force, PieceType.KNIGHT))
    return actions


def cannon_actions_helper(force: Force, pos, new_pos: int, status: CannonMoveStatus, board: Board) -> (
        Action, int, bool):
    if new_pos != -1:  # 棋盘内区域
        if status == CannonMoveStatus.DIRECT:  # direct move status.
            if board.get_force(new_pos) == Force.EMPTY:
                return Action(pos, new_pos, force, PieceType.CANNON), CannonMoveStatus.DIRECT
            else:
                return None, CannonMoveStatus.EMMIT
        elif status == CannonMoveStatus.EMMIT:  # emmit move status.
            if board.get_force(new_pos) == Force.EMPTY:
                return None, CannonMoveStatus.EMMIT
            elif board.get_force(new_pos) != force:
                return Action(pos, new_pos, force, PieceType.CANNON), CannonMoveStatus.FINNISH
            else:
                return None, CannonMoveStatus.FINNISH
    else:  # 棋盘外区域
        return None, CannonMoveStatus.FINNISH


def cannon_actions(force: Force, pos: int, board: Board) -> []:
    actions = []
    status = CannonMoveStatus.DIRECT
    for i in range(1, 9, 1):
        new_pos = board.cal_pos_by_inc(pos, i, 0)
        action, status = cannon_actions_helper(
            force, pos, new_pos, status, board)
        # Board.print(pos, new_pos, action is not None, status)
        if action is not None:
            actions.append(action)
        if status is CannonMoveStatus.FINNISH:
            break
    status = CannonMoveStatus.DIRECT
    for i in range(-1, -9, -1):
        new_pos = board.cal_pos_by_inc(pos, i, 0)
        action, status = cannon_actions_helper(
            force, pos, new_pos, status, board)
        # Board.print(pos, new_pos, action is not None, status)
        if action is not None:
            actions.append(action)
        if status is CannonMoveStatus.FINNISH:
            break
    status = CannonMoveStatus.DIRECT
    for i in range(1, 10, 1):
        new_pos = board.cal_pos_by_inc(pos, 0, i)
        action, status = cannon_actions_helper(
            force, pos, new_pos, status, board)
        # Board.print(pos, new_pos, action is not None, status)
        if action is not None:
            actions.append(action)
        if status is CannonMoveStatus.FINNISH:
            break
    status = CannonMoveStatus.DIRECT
    for i in range(-1, -10, -1):
        new_pos = board.cal_pos_by_inc(pos, 0, i)
        action, status = cannon_actions_helper(
            force, pos, new_pos, status, board)
        # Board.print(pos, new_pos, action is not None, status)
        if action is not None:
            actions.append(action)
        if status is CannonMoveStatus.FINNISH:
            break
    return actions


def bishop_actions(force, pos, board):
    actions = []
    if force == Force.RED:
        if pos == 2:
            if board.empty_pos(10) and board.available_pos(18, Force.RED):
                actions.append(Action(2, 18, force, PieceType.BISHOP))
            if board.empty_pos(12) and board.available_pos(22, Force.RED):
                actions.append(Action(2, 22, force, PieceType.BISHOP))
        elif pos == 6:
            if board.get_force(14) == Force.EMPTY and board.available_pos(22, Force.RED):
                actions.append(Action(6, 22, force, PieceType.BISHOP))
            if board.get_force(16) == Force.EMPTY and board.available_pos(26, Force.RED):
                actions.append(Action(6, 26, force, PieceType.BISHOP))
        elif pos == 18:
            if board.get_force(10) == Force.EMPTY and board.available_pos(2, Force.RED):
                actions.append(Action(18, 2, force, PieceType.BISHOP))
            if board.get_force(28) == Force.EMPTY and board.available_pos(38, Force.RED):
                actions.append(Action(18, 38, force, PieceType.BISHOP))
        elif pos == 22:
            if board.get_force(12) == Force.EMPTY and board.available_pos(2, Force.RED):
                actions.append(Action(22, 2, force, PieceType.BISHOP))
            if board.get_force(14) == Force.EMPTY and board.available_pos(6, Force.RED):
                actions.append(Action(22, 6, force, PieceType.BISHOP))
            if board.get_force(30) == Force.EMPTY and board.available_pos(38, Force.RED):
                actions.append(Action(22, 38, force, PieceType.BISHOP))
            if board.get_force(32) == Force.EMPTY and board.available_pos(42, Force.RED):
                actions.append(Action(22, 42, force, PieceType.BISHOP))
        elif pos == 26:
            if board.get_force(16) == Force.EMPTY and board.available_pos(6, Force.RED):
                actions.append(Action(26, 6, force, PieceType.BISHOP))
            if board.get_force(34) == Force.EMPTY and board.available_pos(42, Force.RED):
                actions.append(Action(26, 42, force, PieceType.BISHOP))
        elif pos == 38:
            if board.get_force(28) == Force.EMPTY and board.available_pos(18, Force.RED):
                actions.append(Action(38, 18, force, PieceType.BISHOP))
            if board.get_force(30) == Force.EMPTY and board.available_pos(22, Force.RED):
                actions.append(Action(38, 22, force, PieceType.BISHOP))
        elif pos == 42:
            if board.get_force(32) == Force.EMPTY and board.available_pos(22, Force.RED):
                actions.append(Action(42, 22, force, PieceType.BISHOP))
            if board.get_force(34) == Force.EMPTY and board.available_pos(26, Force.RED):
                actions.append(Action(42, 26, force, PieceType.BISHOP))
    elif force == Force.BLACK:
        if pos == 87:
            if board.get_force(79) == Force.EMPTY and board.available_pos(71, Force.BLACK):
                actions.append(Action(87, 71, force, PieceType.BISHOP))
            if board.get_force(77) == Force.EMPTY and board.available_pos(67, Force.BLACK):
                actions.append(Action(87, 67, force, PieceType.BISHOP))
        elif pos == 83:
            if board.get_force(75) == Force.EMPTY and board.available_pos(67, Force.BLACK):
                actions.append(Action(83, 67, force, PieceType.BISHOP))
            if board.get_force(73) == Force.EMPTY and board.available_pos(63, Force.BLACK):
                actions.append(Action(83, 63, force, PieceType.BISHOP))
        elif pos == 71:
            if board.get_force(79) == Force.EMPTY and board.available_pos(87, Force.BLACK):
                actions.append(Action(71, 87, force, PieceType.BISHOP))
            if board.get_force(71) == Force.EMPTY and board.available_pos(51, Force.BLACK):
                actions.append(Action(71, 51, force, PieceType.BISHOP))
        elif pos == 67:
            if board.get_force(77) == Force.EMPTY and board.available_pos(87, Force.BLACK):
                actions.append(Action(67, 87, force, PieceType.BISHOP))
            if board.get_force(75) == Force.EMPTY and board.available_pos(83, Force.BLACK):
                actions.append(Action(67, 83, force, PieceType.BISHOP))
            if board.get_force(59) == Force.EMPTY and board.available_pos(51, Force.BLACK):
                actions.append(Action(67, 51, force, PieceType.BISHOP))
            if board.get_force(57) == Force.EMPTY and board.available_pos(47, Force.BLACK):
                actions.append(Action(67, 47, force, PieceType.BISHOP))
        elif pos == 63:
            if board.get_force(73) == Force.EMPTY and board.available_pos(83, Force.BLACK):
                actions.append(Action(63, 83, force, PieceType.BISHOP))
            if board.get_force(55) == Force.EMPTY and board.available_pos(47, Force.BLACK):
                actions.append(Action(63, 47, force, PieceType.BISHOP))
        elif pos == 51:
            if board.get_force(61) == Force.EMPTY and board.available_pos(71, Force.BLACK):
                actions.append(Action(51, 71, force, PieceType.BISHOP))
            if board.get_force(59) == Force.EMPTY and board.available_pos(67, Force.BLACK):
                actions.append(Action(51, 67, force, PieceType.BISHOP))
        elif pos == 47:
            if board.empty_pos(57) and board.available_pos(67, Force.BLACK):
                actions.append(Action(47, 67, force, PieceType.BISHOP))
            if board.empty_pos(55) and board.available_pos(63, Force.BLACK):
                actions.append(Action(47, 63, force, PieceType.BISHOP))
    return actions


def guard_actions(force: Force, pos: int, board: Board) -> []:
    actions = []
    if force == Force.RED:
        if pos == 3 and board.available_pos(13, Force.RED):
            return [Action(3, 13, force, PieceType.GUARD)]
        elif pos == 5 and board.available_pos(13, Force.RED):
            return [Action(5, 13, force, PieceType.GUARD)]
        elif pos == 21 and board.available_pos(13, Force.RED):
            return [Action(21, 13, force, PieceType.GUARD)]
        elif pos == 23 and board.available_pos(13, Force.RED):
            return [Action(23, 13, force, PieceType.GUARD)]
        elif pos == 13:
            if board.available_pos(3, Force.RED):
                actions.append(Action(13, 3, force, PieceType.GUARD))
            if board.available_pos(5, Force.RED):
                actions.append(Action(13, 5, force, PieceType.GUARD))
            if board.available_pos(21, Force.RED):
                actions.append(Action(13, 21, force, PieceType.GUARD))
            if board.available_pos(23, Force.RED):
                actions.append(Action(13, 23, force, PieceType.GUARD))
    elif force == Force.BLACK:
        if pos == 86 and board.available_pos(76, Force.BLACK):
            return [Action(86, 76, force, PieceType.GUARD)]
        elif pos == 84 and board.available_pos(76, Force.BLACK):
            return [Action(84, 76, force, PieceType.GUARD)]
        elif pos == 68 and board.available_pos(76, Force.BLACK):
            return [Action(68, 76, force, PieceType.GUARD)]
        elif pos == 66 and board.available_pos(76, Force.BLACK):
            return [Action(66, 76, force, PieceType.GUARD)]
        elif pos == 76:
            if board.available_pos(86, Force.BLACK):
                actions.append(Action(76, 86, force, PieceType.GUARD))
            if board.available_pos(84, Force.BLACK):
                actions.append(Action(76, 84, force, PieceType.GUARD))
            if board.available_pos(68, Force.BLACK):
                actions.append(Action(76, 68, force, PieceType.GUARD))
            if board.available_pos(66, Force.BLACK):
                actions.append(Action(76, 66, force, PieceType.GUARD))
    return actions


def pawn_actions_helper(force: Force, pos: int, pos_inc: [], board: Board) -> []:
    actions = []
    for (x, y) in pos_inc:
        new_pos = board.cal_pos_by_inc(pos, x, y)
        if new_pos != -1 and board.available_pos(new_pos, force):
            actions.append(Action(pos, new_pos, force, PieceType.PAWN))
    return actions


def pawn_actions(force: Force, pos: int, board: Board) -> []:
    if force == Force.RED:
        if pos <= RED_LIMIT:
            return pawn_actions_helper(force, pos, [(0, 1)], board)
        else:
            return pawn_actions_helper(force, pos, [(0, 1), (-1, 0), (1, 0)], board)
    elif force == Force.BLACK:
        if pos >= BLACK_LIMIT:
            return pawn_actions_helper(force, pos, [(0, -1)], board)
        else:
            return pawn_actions_helper(force, pos, [(0, -1), (-1, 0), (1, 0)], board)
    else:
        return []


if __name__ == "__main__":
    data = [0, 23, 24, 16, 0, 12, 15, 0, 0, 0, 0, 0, 11, 16, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 14, 0, 0, 13, 0, 27, 0, 0, 0, 0, 0, 0, 12, 0, 15, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 17, 14, 27, 0, 17, 0, 0, 27, 0, 0, 0, 0, 0, 27, 0, 0, 0, 26, 25, 0, 24, 0, 25, 0, 0, 22, 0, 0, 21, 0, 0, 0, 0, 0, 23, 26, 0, 22, 0, 0, 0]
    board = Board(data)
    board.show()
    actions = available_actions_with_check(Force.RED, board)
    # actions = available_actions_with_check(Force.BLACK, board)
    # print(check_action(Force.RED, actions[0], board))
    actions = sorted(actions)
    if len(actions) == 0:
        print("no available actions")
    for action in actions:
        piece_from = Board.pos2xy(action.piece_from)
        piece_to = Board.pos2xy(action.piece_to)
        print("from: (%d, %d), to: (%d, %d), force: %s, type: %s" %
              (piece_from[0], piece_from[1], piece_to[0], piece_to[1],
               "红" if action.force == Force.RED else "黑",
               Piece.get_name_by_type(action.piece_type)))
