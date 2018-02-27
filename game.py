

class GameState:

    def is_terminal(self):
        """
        :return: A boolean indicating if the game should continue after this state
        """
        raise Exception("Unimplemented!")

    def get_possible_moves(self):
        """
        :return: A list of legal moves in this state
        """
        raise Exception("Unimplemented!")

    def do_move(self, move):
        """
        Apply a move on the state
        :param move: The move to be applied
        :return: the game state
        """
        raise Exception("Unimplemented!")

    def get_scores(self):
        raise Exception("Unimplemented!")

    def get_score(self):
        """
        :return: 1 if player 1 has won, -1 if player 2 has won, 0 otherwise
        """
        raise Exception("Unimplemented!")

    def get_observation(self):
        """
        :return: a simplified game state from the perspective of the current player
        """
        raise Exception("Unimplemented!")
