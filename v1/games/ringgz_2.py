import numpy as np

from v1.games.ringgz import Ringgz, MAX_X, MAX_Y, SIZES


class Ringgz2(Ringgz):

    def __init__(self):
        super().__init__(2)

    @staticmethod
    def _get_sign(player: int):
        return 1 if player == 0 else -1

    @staticmethod
    def _get_opponent(player: int):
        return 1 - player

    def get_score(self):
        """
        :return: 1 if player 1 has won, -1 if player 2 has won, 0 otherwise
        """
        [m1, m2] = self.get_majorities()
        if m1 < m2:
            return -1
        elif m1 > m2:
            return 1
        else:
            return 0

    def get_scores(self):
        [m1, m2] = self.get_majorities()
        if m1 < m2:
            return [1, 0]
        elif m1 > m2:
            return [0, 1]
        else:
            return [0, 0]

    def get_reward(self):
        """
        :return: 1 if the current player won, -1 if the current player lost, 0 otherwise
        """
        [m1, m2] = self.get_majorities()

        if m1 < m2:
            return -1 * self._get_sign(self.current_player)
        elif m1 > m2:
            return self._get_sign(self.current_player)
        else:
            return 0

    def get_observation(self):
        """
        :return: a simplified game state from the perspective of the current player
        """
        # Initialize observation
        field = np.zeros(shape=(MAX_X,      # X Coordinates
                                MAX_Y,      # Y Coordinates
                                SIZES + 1,  # Ring + Base locations
                                2,          # Two colors
                                2           # For each player sign
                                ))
        # Obtain current player perspective
        current_colors = self.player_colors[self.current_player]
        opponent_colors = self.player_colors[self._get_opponent(self.current_player)]
        # Fill values
        for x in range(MAX_X):
            for y in range(MAX_Y):
                territory = self.territories[x][y]
                # Skip empty territories
                if territory is None or territory == ([None] * SIZES):
                    continue
                # Check for base
                if isinstance(territory, tuple):
                    c = self._get_color_of_piece(territory)
                    if c in current_colors:
                        field[x][y][SIZES][current_colors.index(c)] = 1
                    else:
                        field[x][y][SIZES][opponent_colors.index(c)] = -1
                # Check for rings
                elif isinstance(territory, list):
                    for s in range(SIZES):
                        # Check if there is a ring in this slot
                        if territory[s] is not None:
                            c = self._get_color_of_piece(territory[s])
                            if c in current_colors:
                                field[x][y][s][current_colors.index(c)] = 1
                            else:
                                field[x][y][s][opponent_colors.index(c)] = -1
        return field

