import numpy as np
from game import GameState

# Grid boundaries
MAX_X, MAX_Y = (7, 6)
# Number of players
NUMBER_OF_PLAYERS = 2
# Number of pieces that need to align
CONNECT = 4


class Connect4(GameState):

    def __init__(self):
        self.grid = np.zeros((MAX_X, MAX_Y))
        self.current_player = 1
        self.hash = 0
        self.winner = 0
        self.last_move = None
        self.possible_moves = self._compute_possible_moves()

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
        return int(self.hash)

    @staticmethod
    def valid_x(x: int):
        return 0 <= x < MAX_X

    @staticmethod
    def valid_y(y: int):
        return 0 <= y < MAX_Y

    @staticmethod
    def valid_coordinate(x: int, y: int):
        return 0 <= x < MAX_X and 0 <= y < MAX_Y

    def get_possible_moves(self):
        return self.possible_moves

    def _compute_possible_moves(self):
        return [x for x in range(MAX_X) if self.grid[x][MAX_Y - 1] == 0]  # TODO -- complement top row

    def _update_possible_moves(self):
        self.possible_moves = self._compute_possible_moves()

    def is_terminal(self):
        return self.winner != 0 or len(self.possible_moves) == 0

    def do_move(self, x: int):
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
        count = 0
        for x_, y_ in [(x + i * dx, y + i * dy) for i in range(1, CONNECT)]:
            if self.valid_coordinate(x_, y_) and self.grid[x_][y_] == self.current_player:
                count += 1
            else:
                return count
        return count

    def _switch_players(self):
        self.current_player *= -1
        return self.current_player

    @staticmethod
    def _player_to_index(player: int):
        assert abs(player) == 1
        if player == 1:
            return 0
        else:
            return 1

    def get_scores(self):
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
        return list(range(MAX_X))

    @staticmethod
    def get_observation_size():
        return MAX_X


if __name__ == '__main__':

    state = Connect4()

    while not state.is_terminal():
        moves = state.get_possible_moves()
        move = moves[np.random.choice(len(moves))]
        state.do_move(move)

        print(state)








