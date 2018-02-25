

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
        """
        :return: The scores for each player
        """
        raise Exception("Unimplemented!")

    def get_score(self):
        """
        :return: The score of the player that has just made a move
        """
        raise Exception("Unimplemented!")

    def get_observation(self):
        """
        :return: A suitable neural network input
        """
        raise Exception("Unimplemented!")
