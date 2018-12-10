import numpy as np

from v2.model import Model
from v2.montecarlo import MCST


class Pit:
    """
        Two models enter the pit to play a number of games to see which model is better
    """

    def __init__(self, m1: Model, m2: Model, game_setup: callable, num_duels: int, num_sims: int):
        """
        Create a new pit
        :param m1: Model 1
        :param m2: Model 2
        :param game_setup: Function that returns the initial game state of the game to be played
        :param num_duels: The number of games that the two models play
        :param num_sims: The number of Monte Carlo searches each model gets before choosing an action
        """
        self.wins = np.zeros(2)  # Keep track of the number of games won for each model
        self.m1, self.m2 = m1, m2
        self.mcst1, self.mcst2 = MCST(), MCST()
        self.game_setup = game_setup
        self.num_duels = num_duels
        self.num_sims = num_sims
        self.terminal = False

    def play(self):
        """
        Let the two models play duels against each other
        :return: The ratio of duels that was won by model 1
        """
        if self.terminal:
            raise Exception("This pit has already been played!")

        # Let the models play a number of games
        for duel in range(self.num_duels):
            # Initialize a new game
            state = self.game_setup()
            current_player = 0

            # Store which model corresponds to which player
            # Let the models take turns in who is the starting player
            models = {duel % 2: (self.m1, self.mcst1),
                      (duel + 1) % 2: (self.m2, self.mcst2)}

            # Play the game
            while not state.is_terminal():
                model, tree = models[current_player]
                # Perform a number of Monte Carlo searches
                for _ in range(self.num_sims):
                    tree.search(state, model)
                # Determine an action by sampling from the policy as defined by the tree
                a, _ = tree.action(state, temperature=0)
                # Perform the move
                state.do_move(a)
                current_player = 1 - current_player
            # Add the game result to the win counter (taking player perspective into account)
            if duel % 2 == 0:
                self.wins += state.get_scores()
            else:
                self.wins += np.roll(state.get_scores(), 1)
        self.terminal = True
        # Return the fraction of games won by model 1
        return self.m1_win_freq()

    def m1_win_freq(self):
        """
        :return: The ratio of duels that was won by model 1
        """
        return self.wins[0] / sum(self.wins) if sum(self.wins) != 0 else 0.5


if __name__ == '__main__':
    from v2.tictactoe import TicTacToe
    from v2.tictactoe_model import TicTacToeModel

    _m1, _m2 = TicTacToeModel(), TicTacToeModel()

    _p = Pit(_m1, _m2, TicTacToe, num_sims=40, num_duels=100)

    _r = _p.play()
