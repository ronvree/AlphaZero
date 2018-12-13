from copy import deepcopy

from v2.model import Model
from v2.montecarlo import MCST


class Episode:
    """
        One episode of self-play where one model generates the moves for the perspectives of both players.
        The progression of the game is stored as examples to learn from. Each example is a 3-tuple
        consisting of: - The game state from the moving player's perspective
                       - The policy obtained from the search tree
                       - A value indicating whether the current player won the game (so from
                         the moving player's perspective for each example)
    """

    def __init__(self, game_setup: callable, model: Model, num_sims: int, num_expl: int):
        """
        Create and execute an episode of self-play
        :param game_setup: Function that returns the initial state of the game to be played
        :param model: The model that will be used to generate examples
        :param num_sims: Number of Monte Carlo searches executed before selecting an action
        :param num_expl: Number of initial exploring moves (with tau=1 instead of tau=0)
        """
        self.game_setup = game_setup
        self.model = model
        self.examples = []
        self.mcst = MCST()

        # Initialize the game state
        state = self.game_setup()
        move_counter = 0
        while not state.is_terminal():
            print(state)
            # Perform a number of Monte Carlo Tree Search simulations
            # self.mcst.clear()
            for _ in range(num_sims):
                self.mcst.search(state, model)

            # The first moves have a temperature of 1 for increased exploration
            tau = 1 if move_counter < num_expl else 0

            # Obtain an improved policy from the tree
            a, pi = self.mcst.action(state, temperature=tau)

            # Store examples for the model to train on. One example consists of:
            # - The game state from the current player's perspective
            # - A policy obtained from the search tree
            # - A value indicating if the player eventually won the game (so is added later)
            self.examples.append([deepcopy(state), pi])

            # Perform the move
            state.do_move(a)
            move_counter += 1

        print(state)

        # If the game ended, add the final result to the examples and return them
        # reward=1 if player 1 won, reward=-1 if player 2 won
        reward = state.get_score()
        for e in self.examples:
            e.append(reward)
            reward *= -1  # Switch player perspective

    def get_examples(self):
        """
        :return: the examples generated in this episode
        """
        return self.examples


if __name__ == '__main__':
    from v2.tictactoe import TicTacToe
    from v2.tictactoe_model import TicTacToeModel

    _m = TicTacToeModel()

    _e = Episode(TicTacToe, TicTacToeModel(), num_sims=3, num_expl=3)

    [print(_ex) for _ex in _e.get_examples()]
