

class GameState:
    """
        Interface for games that are suitable for AlphaZero

        Games should be fully observable and have a finite action space

    """

    @staticmethod
    def action_space() -> list:
        """
        :return: A list of all possible actions that could be performed in the game
        """
        raise NotImplementedError

    @staticmethod
    def board_shape() -> tuple:
        """
        :return: A tuple with the shape of the board
        """
        raise NotImplementedError

    def is_terminal(self) -> bool:
        """
        :return: A boolean indicating if the game should continue after this state
        """
        raise NotImplementedError

    def get_actions(self) -> list:
        """
        :return: A list of legal moves in this state
        """
        raise NotImplementedError

    def do_move(self, move):
        """
        Apply a move on the state
        :param move: The move to be applied
        :return: the game state
        """
        raise NotImplementedError

    def get_scores(self):
        """
        :return: [1, 0] if player 1 won,
                 [0, 1] if player 2 won,
                 [0, 0] otherwise
        """
        raise NotImplementedError

    def get_score(self):
        """
        :return: 1 if player 1 has won,
                -1 if player 2 has won,
                 0 otherwise
        """
        raise NotImplementedError

    def get_reward(self):
        """
        :return: 1 if the current player won,
                -1 if the current player lost,
                 0 otherwise
        """
        raise NotImplementedError
