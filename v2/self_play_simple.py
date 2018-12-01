import numpy as np

from collections import deque
from tqdm import tqdm

from v2.episode import Episode
from v2.game import GameState
from v2.model import Model
from v2.config import *

"""
    Alternative self play implementation where during the comparison between two models the actions are sampled from
    the distribution output p of the neural network directly, instead of the distribution pi obtained from the Monte
    Carlo search. This is necessary to be able to train on trivial games (such as TicTacToe), as Monte Carlo search is
    on its own sufficient to give a good policy regardless of the neural network weights. To compare the quality of the
    network output it is then required to directly sample from p. 
"""


class SimplePit:

    def __init__(self, m1: Model, m2: Model, game_setup: callable, num_duels: int):
        """
        Alternative pit class (see v2.duel.Pit) in which the actions are sampled directly from the neural network policy
        output
        :param m1: Model 1
        :param m2: Model 2
        :param game_setup: Function that returns the initial state of the game to be played
        :param num_duels: Number of games that will be played
        """
        self.wins = np.zeros(2)
        self.m1, self.m2 = m1, m2
        self.game_setup = game_setup
        self.num_duels = num_duels
        self.terminal = False

    def play(self):
        """
        Let the two models play duels against each other
        :return: The ratio of duels that was won by model 1
        """
        if self.terminal:
            raise Exception("This put has already been played!")

        # Let the models play a number of games
        for duel in range(self.num_duels):
            # Initialize a new game
            state = self.game_setup()
            current_player = 0

            # Store which model corresponds to which player
            # Let the models take turns in who is the starting player
            models = {duel % 2: self.m1,
                      (duel + 1) % 2: self.m2}

            # Play the game
            while not state.is_terminal():
                model = models[current_player]

                # Determine an action by sampling from the output policy of the model
                a = sample_action_from_model(model, state)

                # Perform the move
                state.do_move(a)
                current_player = 1 - current_player

            # Add the game results to the win counter (taking player perspective into account)
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


def policy_iter_self_play(game_setup: callable,
                          model: Model,
                          num_iter: int=NUM_ITER,
                          num_epis: int=NUM_EPIS,
                          num_exam: int=NUM_EXAM,
                          num_sims: int=NUM_SIMS_EPIS,
                          num_duel: int=NUM_DUEL,
                          num_expl: int=NUM_EXPL,
                          win_threshold: int=WIN_THRESHOLD):
    """
    Perform policy iteration to train the model
    :param game_setup: Function that returns the initial state of the game to be played
    :param model: The model to be trained
    :param num_iter: The number of iterations that the policy iteration runs
    :param num_epis: The number of episodes that are used to generate example in each iteration of self-play
    :param num_exam: The number of most recent examples that are used to train the model
    :param num_sims: The number of Monte Carlo searches performed before selecting a move when playing an episode
    :param num_duel: The number of duels performed to select the best of two competing models
    :param num_expl: The number of initial exploring moves (with tau=1 instead of tau=0) when playing an episode
    :param win_threshold: The ratio of wins that a model needs to achieve in the pit to be considered superior
    :return: the trained model
    """
    examples = deque(maxlen=num_exam)
    model_counter = 0  # TODO -- remove
    iter_counter = 0  # TODO -- remove, better logging
    for i in range(num_iter):
        # Play a number of games of self-play and obtain examples to learn from
        # Each example is a tuple of (game state, corresponding policy, final reward)
        for _ in tqdm(range(num_epis)):
            examples += Episode(game_setup,
                                model,
                                num_sims=num_sims,
                                num_expl=num_expl).get_examples()
        # Train a candidate new model on all examples
        candidate_model = model.fit_new_model(examples)
        # Let the model and its candidate compete for a number of games
        frac_win = SimplePit(candidate_model,
                             model,
                             game_setup,
                             num_duels=num_duel).play()
        # If the candidate model won more than the set threshold, reject the old model
        if frac_win > win_threshold:
            model = candidate_model
            model_counter += 1
        print('Fraction of games won: {}'.format(frac_win))
        print('Models replaced so far: {}'.format(model_counter))
        print('Iteration {} complete'.format(iter_counter))
        iter_counter += 1
    return model


def sample_action_from_model(model: Model, state: GameState, take_max: bool=False):
    """
    Sample an action from the output of the given model
    :param model: Model from whose output the action will be sampled
    :param state: The state that will form the input for the model
    :param take_max: A boolean indicating whether the action should be sampled greedily (that is, always take highest
                     probability)
    :return: The sampled action
    """
    p, _ = model.predict(state)  # Obtain policy p from the model

    actions = state.get_actions()  # Get the legal actions for this game state

    # Normalize the distribution to only include valid actions
    valid_a_dist = {a: p[a] for a in actions}
    a_dist = {a: valid_a_dist[a] / sum(valid_a_dist.values()) for a in actions}

    # Sample an action from the distribution
    if take_max:
        a = max(a_dist, key=a_dist.get)
    else:
        items = list(a_dist.items())
        a = items[np.random.choice(len(items), p=[p[1] for p in items])][0]
    return a


if __name__ == '__main__':
    from v2.tictactoe import TicTacToe
    from v2.tictactoe_model import TestNetwork

    _m = TestNetwork()

    _m.model.load_weights('checkpoint.pth.tar')

    _m = policy_iter_self_play(TicTacToe,
                               _m,
                               num_iter=100,
                               num_exam=90000,
                               num_sims=100,
                               num_duel=30,
                               num_expl=3,
                               num_epis=100)
