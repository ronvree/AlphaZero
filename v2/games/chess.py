import numpy as np

from chess import Board, square, square_rank, square_file, Move
from chess import PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING

from itertools import product

from v2.game import GameState

BOARD_SHAPE = MAX_X, MAX_Y = 8, 8

SQUARES = list(product(range(MAX_X), range(MAX_Y)))

ACTIONS = list(product(SQUARES, SQUARES))

PIECE_INDICES = {
    PAWN: 0,
    ROOK: 1,
    KNIGHT: 2,
    BISHOP: 3,
    QUEEN: 4,
    KING: 5
}

NUM_PIECE_TYPE = len(PIECE_INDICES)

# ZOBRIST_TABLE = np.random.randint(1,
#                                   high=(2 ** 63) - 1,
#                                   size=(2, MAX_X, MAX_Y, NUM_PIECE_TYPE),
#                                   dtype=np.int64)


class Chess(GameState):

    def __init__(self):
        """
        Wraps around chess.Board to make it compatible with this AlphaZero implementation
        """
        self.board = Board()
        self.possible_moves = list(self.board.legal_moves)

        # self.hash = 0
        # for sq, p in self.board.piece_map().items():    # TODO -- zobrist hash!!
        #     x, y = Chess.square_index_to_xy(sq)
            # self.hash ^= ZOBRIST_TABLE[p.color][x][y][PIECE_INDICES[p.piece_type]]

    def __hash__(self):
        return hash(' '.join(self.board.fen().split()[:-1]))

    def __str__(self):
        return '\n' + str(self.board) + '\n' + self.board.fen() + '\n'

    @staticmethod
    def action_space() -> list:
        return ACTIONS

    @staticmethod
    def board_shape() -> tuple:
        return BOARD_SHAPE

    @staticmethod
    def square_index_to_xy(i: int) -> tuple:
        return square_file(i), square_rank(i)

    @staticmethod
    def xy_to_square_index(file: int, rank: int):
        return square(file, rank)

    @staticmethod
    def move_to_tuple(move: Move):
        fr_square = Chess.square_index_to_xy(move.from_square)
        to_square = Chess.square_index_to_xy(move.to_square)
        return fr_square, to_square

    @staticmethod
    def select_move_from(move: tuple, moves: list):
        for m in moves:
            if Chess.move_to_tuple(m) == move:
                if m.promotion is not None and m.promotion != 5:
                    continue  # Keep searching until a promotion with queen has been found
                return m
        raise Exception("Could not find {} among moves {}!".format(move, moves))

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def get_actions(self) -> list:
        return [Chess.move_to_tuple(m) for m in self.possible_moves if m.promotion is None or m.promotion == 5]

    def do_move(self, move):
        self.board.push(Chess.select_move_from(move, self.possible_moves))
        self.possible_moves = list(self.board.legal_moves)
        return self

    def get_scores(self):
        r = self._parse_result()
        if r == 1:
            return [1, 0]
        elif r == -1:
            return [0, 1]
        else:
            return [0, 0]

    def get_score(self):
        return self._parse_result()

    def get_reward(self):
        return self._get_current_player() * self.get_score()

    def _get_current_player(self):
        """
        Convert the chess.board current player representation (True/False) to this GameState current
        player representation (1/-1)
        :return: 1 if it is player 1's turn, -1 if it is player 2's turn
        """
        if self.board.turn:
            return 1
        else:
            return -1

    def _parse_result(self):
        r = self.board.result()
        if r == '*':
            return 0
        elif r == '1-0':
            return 1
        elif r == '0-1':
            return -1
        elif r == '1/2-1/2':
            return 0
        else:
            raise Exception("Could not parse chess board result: {} of type {}".format(r, type(r)))

    def get_current_player(self):
        """
        :return: 1 if player 1 has to move
                -1 if player 2 has to move
        """
        if self.board.turn:
            return 1
        else:
            return -1


if __name__ == '__main__':
    import chess

    # from chess import Board
    #
    board = chess.Board()

    print(board)
    print(board.legal_moves)
