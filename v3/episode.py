import argparse
from copy import deepcopy

from v3.model import Model
from v3.montecarlo import MCST


class Episode:
    """
        One episode of self-play where one model generates the moves for the perspectives of both players.
        The progression of the game is stored as examples to learn from. Each example is a 3-tuple
        consisting of: - The game state from the moving player's perspective
                       - The policy obtained from the search tree
                       - A value indicating whether the current player won the game (so from
                         the moving player's perspective for each example)
    """

    def __init__(self,
                 game_setup: callable,
                 model: Model,
                 args: argparse.Namespace
                 ):
        self.game_setup = game_setup
        self.model = model
        self.examples = []
        self.mcst = MCST(args)
        self.log_items = []

        # Initialize the game state
        state = self.game_setup(args)
        move_counter = 0
        while not state.is_terminal():
            # Perform a number of Monte Carlo Tree Search simulations
            for _ in range(args.num_sims_epis):
                self.mcst.search(state, model)

            # The first moves have a temperature of 1 for increased exploration
            tau = 1 if move_counter < args.num_expl else 0

            # Obtain an improved policy from the tree
            a, pi = self.mcst.action(state, temperature=tau)

            # Store examples for the model to train on. One example consists of:
            # - The game state from the current player's perspective
            # - A policy obtained from the search tree
            # - A value indicating if the player eventually won the game (so is added later)
            self.examples.append([deepcopy(state), pi])

            # Store values for logging
            self.log_items.append((deepcopy(state), move_counter, a, pi, tau))

            # Perform the move
            state.do_move(a)
            move_counter += 1

        # If the game ended, add the final result to the examples and return them
        # reward=1 if player 1 won, reward=-1 if player 2 won
        reward = state.get_score()
        for e in self.examples:
            e.append(reward)
            reward *= -1  # Switch player perspective

        # Write episode to log file
        # if args.log_file:  # TODO -- logging properly
        #     with open(args.log_file, 'w') as f:
        #         for s, c, a, pi, tau in self.log_items:
        #             f.write('\n{}\nMove {}\nChose action: {}\nFrom policy: {}\nWith tau: {}\n'.format(
        #                 s, c, a, pi, tau))

    def get_examples(self):
        """
        :return: the examples generated in this episode
        """
        return self.examples


if __name__ == '__main__':
    from v3.games.tictactoe.tictactoe import TicTacToe
    from v3.model import DummyModel

    _args = argparse.Namespace()
    _args.num_sims_epis = 3
    _args.num_expl = 3
    _args.c_puct = 1
    _args.epsilon = 0.25
    _args.alpha = 0.03
    _args.log_file = None

    _m = DummyModel(TicTacToe, _args)

    _e = Episode(TicTacToe, _m, _args)

    [print(_ex) for _ex in _e.get_examples()]
