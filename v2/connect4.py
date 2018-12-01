import numpy as np
from v2.game import GameState

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
    def __init__(self):
        self.grid = np.zeros((MAX_X, MAX_Y))
        self.current_player = 1
        self.hash = 0
        self.winner = 0
        self.last_move = None
        self.possible_moves = self._compute_possible_moves()

    # Initialize a static table used for Zobrist hashing
    _zobrist_table = np.random.randint(1, high=(2 ** 31) - 1, size=(2, MAX_X, MAX_Y))

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
        return [x for x in range(MAX_X) if self.grid[x][MAX_Y - 1] == 0]  # TODO -- complement top row

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
        for y in range(MAX_Y):
            if self.grid[x][y] == 0:
                self.grid[x][y] = self.current_player
                self.hash ^= self._zobrist_table[self._player_to_index(self.current_player)][x][y]
                self.last_move = (x, y)  # TODO -- last move doesnt need to be stored in game state
                break
        self._update_winner()
        self._update_possible_moves()
        self._switch_players()
        return self

    def _update_winner(self):
        """
        Check whether the last move that has been made satisfies the winning criteria. If so, updates game winner
        """
        if self.last_move is not None:
            x, y = self.last_move
            for dx in [0, 1, -1]:  # TODO -- CHECK WIN BY MATRIX INNER PRODUCT
                for dy in [0, 1, -1]:
                    if dx == 0 and dy == 0:  # TODO -- METHOD NOW COMPUTES SOME AXES MULTIPLE TIMES!
                        continue
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

    def get_observation(self):
        """
        :return: a simplified game state from the perspective of the current player
        """
        return self.current_player * self.grid

    @staticmethod
    def get_all_actions():
        """
        :return: All possible actions that can be played in a game of Connect 4
        """
        return list(range(MAX_X))  # TODO -- can be removed right?


if __name__ == '__main__':

    state = Connect4()

    while not state.is_terminal():
        moves = state.get_actions()
        move = moves[np.random.choice(len(moves))]
        state.do_move(move)

        print(state)
