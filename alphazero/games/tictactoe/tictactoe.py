import numpy as np
import argparse

from alphazero.game import GameState

GRID_SHAPE = MAX_X, MAX_Y = 3, 3

NUMBER_OF_PLAYERS = 2

CONNECT = 3

PLAYER_1_SYMBOL, PLAYER_2_SYMBOL = 1, -1


class TicTacToe(GameState):

    def __init__(self, args: argparse.Namespace):
        self.grid = np.zeros(shape=GRID_SHAPE)
        self.current_player = 1
        self.hash = 0
        self.winner = 0
        self.possible_moves = TicTacToe.action_space()
        self.args = args

    _zobrist_table = np.random.randint(1, high=(2 ** 31) - 1, size=(2, MAX_X, MAX_Y))

    def __str__(self):
        s = '+' + ('-' * MAX_X) + '+\n'
        for y in range(MAX_Y - 1, -1, -1):
            s += '|'
            for x in range(MAX_X):
                if self.grid[x][y] == 1:
                    s += 'X'
                elif self.grid[x][y] == -1:
                    s += 'O'
                else:
                    s += ' '
            s += '|\n'
        s += '+' + ('-' * MAX_X) + '+\n'
        return s

    def __hash__(self):
        """
        :return: The pre-computed Zobrist hash
        """
        return int(self.hash)

    @staticmethod
    def action_space():
        actions = []
        for x in range(MAX_X):
            for y in range(MAX_Y):
                actions.append((x, y))
        return actions

    @staticmethod
    def board_shape() -> tuple:
        return MAX_X, MAX_Y

    @staticmethod
    def valid_x(x: int):
        return 0 <= x < MAX_X

    @staticmethod
    def valid_y(y: int):
        return 0 <= y < MAX_Y

    @staticmethod
    def valid_coordinate(x: int, y: int):
        return 0 <= x < MAX_X and 0 <= y < MAX_Y

    @staticmethod
    def _player_to_index(player: int):
        assert abs(player) == 1
        if player == 1:
            return 0
        else:
            return 1

    def get_actions(self):
        return self.possible_moves

    def is_terminal(self):
        return self.winner != 0 or len(self.possible_moves) == 0

    def do_move(self, move: tuple):
        x, y = move
        self.grid[x][y] = self.current_player
        self.hash ^= self._zobrist_table[self._player_to_index(self.current_player)][x][y]
        self.possible_moves.remove(move)
        self._update_winner(move)
        self._switch_players()
        return self

    def _update_winner(self, last_move: tuple):
        """
        Check whether the last move that has been made satisfies the winning criteria. If so, updates game winner
        """
        if last_move is not None:
            x, y = last_move
            for dx, dy in [(0, 1), (1, 1), (1, 0), (1, -1)]:
                connect = 1 + self._count_in_direction(x, y, dx, dy) + self._count_in_direction(x, y, -dx, -dy)
                if connect >= CONNECT:
                    self.winner = self.current_player
                    return

    def _count_in_direction(self, x, y, dx, dy):
        """
        Helper method for self._update_winner. Counts the number of subsequent pieces in the specified direction
        :param x: The x coordinate to start from
        :param y: The y coordinate to start from
        :param dx: The change in x per step in the direction
        :param dy: The change in y per step in the direction
        :return: The number of subsequent pieces equal to the piece in (x, y) in the specified direction (excluding
                 the piece at (x, y)
        """
        count = 0
        for x_, y_ in [(x + i * dx, y + i * dy) for i in range(1, CONNECT)]:
            if self.valid_coordinate(x_, y_) and self.grid[x_][y_] == self.current_player:
                count += 1
            else:
                return count
        return count

    def _switch_players(self):
        self.current_player *= -1

    def get_scores(self):
        scores = [0] * NUMBER_OF_PLAYERS
        if self.winner != 0:
            scores[self._player_to_index(self.winner)] = 1
        return scores

    def get_score(self):
        return self.winner

    def get_reward(self):
        return self.current_player * self.winner

    def get_current_player(self):
        """
        :return: 1 if player 1 has to move
                -1 if player 2 has to move
        """
        return self.current_player


if __name__ == '__main__':
    import argparse

    _args = argparse.Namespace()

    _state = TicTacToe(_args)

    while not _state.is_terminal():
        _moves = _state.get_actions()
        _move = _moves[np.random.choice(len(_moves))]
        _state.do_move(_move)

        print(_state)

    print(_state.get_reward())
    print(_state.get_score())
