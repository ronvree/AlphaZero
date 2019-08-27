import numpy as np
import argparse

from v3.model import Model
from v3.montecarlo import MCST


class Pit:
    """
        Two models enter the pit to play a number of games to see which model is better
    """

    def __init__(self, m1: Model, m2: Model, game: callable, args: argparse.Namespace):
        """
        Create a new pit
        :param m1: Model 1
        :param m2: Model 2
        :param game: Function that returns the initial game state of the game to be played (typically its constructor)
                     Takes 1 argument. Namely the argparse.Namespace used to pass parameters
        :param args: Parsed arguments containing hyperparameters
                        - num_duels: the number of duels played in the pit
                        - num_sims_duel: the number of Monte Carlo searches each model gets before selecting a move
        """
        self.wins = np.zeros(2)
        self.m1, self.m2 = m1, m2
        self.mcst1, self.mcst2 = MCST(args), MCST(args)
        self.game_setup = game
        self.num_duels = args.num_duels
        self.num_sims = args.num_sims_duel
        self.terminal = False
        self.args = args

    def play(self):  # TODO -- batches of games
        """
        Let the two models play duels against each other
        :return: A list containing the number of wins by model 1 and model 2, respectively
        """
        if self.terminal:
            raise Exception("This pit has already been played!")

        # Let the models play a number of games
        for duel in range(self.num_duels):
            # Initialize a new game
            state = self.game_setup(self.args)
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

        return self.wins


if __name__ == '__main__':
    from v3.games.tictactoe.tictactoe import TicTacToe
    from v3.model import DummyModel

    _args = argparse.Namespace()
    _args.num_sims_duel = 40
    _args.num_duels = 100
    _args.epsilon = 0.25
    _args.alpha = 0.03
    _args.c_puct = 1

    _m1, _m2 = DummyModel(TicTacToe, _args), DummyModel(TicTacToe, _args)

    _p = Pit(_m1, _m2, TicTacToe, _args)

    _r = _p.play()

    print(_r)
