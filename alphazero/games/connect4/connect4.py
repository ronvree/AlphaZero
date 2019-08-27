import numpy as np

import argparse

from alphazero.game import GameState

# Grid boundaries
MAX_X, MAX_Y = (7, 6)
# Number of players
NUMBER_OF_PLAYERS = 2
# Number of pieces that need to align
CONNECT = 4


class Connect4(GameState):
    """
    A game state class corresponding to the game of Connect 4
    """

    def __init__(self, args: argparse.Namespace):
        self.grid = np.zeros((MAX_X, MAX_Y))
        self.current_player = 1
        self.hash = 0
        self.winner = 0
        self.possible_moves = self._compute_possible_moves()
        self.args = args

    # Initialize a static table used for Zobrist hashing
    _zobrist_table = np.random.randint(1, high=(2 ** 63) - 1, size=(2, MAX_X, MAX_Y), dtype=np.int64)

    def __str__(self):
        """
        :return: A pretty representation of the game state
        """
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
        return list(range(MAX_X))

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

    def get_actions(self):
        """
        :return: The pre-computed valid actions in this game state
        """
        return self.possible_moves

    def _compute_possible_moves(self):
        """
        :return: A list of all valid actions in this game state
        """
        return [x for x in range(MAX_X) if self.grid[x][MAX_Y - 1] == 0]

    def _update_possible_moves(self):
        """
        Compute and set the valid moves in this game state
        """
        self.possible_moves = self._compute_possible_moves()

    def is_terminal(self):
        """
        :return: A boolean indicating whether the game has ended
        """
        return self.winner != 0 or len(self.possible_moves) == 0

    def do_move(self, x: int):
        """
        Perform a move on this game state
        :param x: The move to be performed
        :return: self, after the move has been performed
        """
        assert self.valid_x(x)
        xy = None
        for y in range(MAX_Y):
            if self.grid[x][y] == 0:
                self.grid[x][y] = self.current_player
                self.hash ^= self._zobrist_table[self._player_to_index(self.current_player)][x][y]
                xy = (x, y)
                break
        assert xy is not None
        self._update_winner(xy)
        self._update_possible_moves()
        self._switch_players()
        return self

    def _update_winner(self, last_move):
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
        """
        Switch which player's turn it is
        :return: The current player after switching
        """
        self.current_player *= -1
        return self.current_player

    @staticmethod
    def _player_to_index(player: int):
        """
        Map both player 1 and 2 to index 0 and 1, respectively
        :param player: The player for which the index should be obtained. Note that this is 1 for player 1 and -1 for
                       player 2
        :return:
        """
        assert abs(player) == 1
        if player == 1:
            return 0
        else:
            return 1

    def get_scores(self):
        """
        :return: [1, 0] if player 1 won,
                 [0, 1] if player 2 won,
                 [0, 0] otherwise
        """
        scores = [0] * NUMBER_OF_PLAYERS
        if self.winner != 0:
            scores[self._player_to_index(self.winner)] = 1
        return scores

    def get_score(self):
        """
        :return: 1 if player 1 has won, -1 if player 2 has won, 0 otherwise
        """
        return self.winner

    def get_reward(self):
        """
        :return: 1 if the current player won, -1 if the current player lost, 0 otherwise
        """
        return self.current_player * self.winner

    def get_current_player(self):
        """
        :return: 1 if player 1 has to move
                -1 if player 2 has to move
        """
        return self.current_player


if __name__ == '__main__':

    state = Connect4(argparse.Namespace())

    while not state.is_terminal():
        moves = state.get_actions()
        move = moves[np.random.choice(len(moves))]
        state.do_move(move)

        print(state)
