from collections import deque

from tqdm import tqdm

from v2.duel import Pit
from v2.episode import Episode
from v2.model import Model
from v2.config import *

"""
    Train the model through policy iteration
    
    TODO -- Better logging
    
"""


def policy_iter_self_play(game_setup: callable,
                          model: Model,
                          num_iter: int = NUM_ITER,
                          num_epis: int = NUM_EPIS,
                          num_exam: int = NUM_EXAM,
                          num_sims_epis: int = NUM_SIMS_EPIS,
                          num_sims_duel: int = NUM_SIMS_DUEL,
                          num_duel: int = NUM_DUEL,
                          num_expl: int = NUM_EXPL,
                          win_threshold: int = WIN_THRESHOLD):
    """
    Perform policy iteration to train the model
    :param game_setup: Function that returns the initial state of the game to be played
    :param model: The model to be trained
    :param num_iter: The number of iterations that the policy iteration runs
    :param num_epis: The number of episodes that are used to generate example in each iteration of self-play
    :param num_exam: The number of most recent examples that are used to train the model
    :param num_sims_epis: The number of Monte Carlo searches performed before selecting a move when playing an episode
    :param num_sims_duel: The number of Monte Carlo searches performed before selecting a move when comparing two models
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
                                num_sims=num_sims_epis,
                                num_expl=num_expl).get_examples()
        # Train a candidate new model on all examples
        candidate_model = model.fit_new_model(examples)
        # Let the model and its candidate compete for a number of games
        frac_win = Pit(candidate_model,
                       model,
                       game_setup,
                       num_duels=num_duel,
                       num_sims=num_sims_duel).play()
        # If the candidate model won more than the set threshold, reject the old model
        if frac_win > win_threshold:
            model = candidate_model
            model_counter += 1
        print('Fraction of games won: {}'.format(frac_win))
        print('Models replaced so far: {}'.format(model_counter))
        print('Iteration {} complete'.format(iter_counter))
        iter_counter += 1
    return model


if __name__ == '__main__':
    from v2.connect4 import Connect4
    from v2.connect4_model import Connect4Model

    _m = Connect4Model()

    _m = policy_iter_self_play(Connect4, _m)
