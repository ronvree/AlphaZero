import os
from collections import deque

from tqdm import tqdm

from v2.duel import Pit
from v2.episode import Episode
from v2.model import Model
from v2.config import *

"""
    Train the model through policy iteration
    
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
                          win_threshold: int = WIN_THRESHOLD,
                          log_dir: str = None
                          ):
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
    :param log_dir: The directory where the self play progress should be logged. Set to None by default (for no logging)
    :return: the trained model
    """
    if log_dir and not os.path.exists(log_dir):
        os.mkdir(log_dir)
    iter_log = log_dir + '\\iter_log.txt' if log_dir is not None else None
    epis_log_dir = log_dir + '\\iter_{}' if log_dir is not None else None
    epis_log_file = '\\episode_{}.txt' if log_dir is not None else None

    examples = deque(maxlen=num_exam)
    model_counter = 0  # Count number of models that have been replaced so far
    for i in range(num_iter):
        # Prepare directory for logging
        epis_log_dir_i = epis_log_dir.format(i) if epis_log_dir is not None else None
        if epis_log_dir_i and not os.path.exists(epis_log_dir):
            os.mkdir(epis_log_dir_i)

        # Play a number of games of self-play and obtain examples to learn from
        # Each example is a tuple of (game state, corresponding policy, final reward)
        for j in tqdm(range(num_epis)):
            examples += Episode(game_setup,
                                model,
                                num_sims=num_sims_epis,
                                num_expl=num_expl,
                                log_file=(epis_log_dir + epis_log_file).format(i, j),
                                ).get_examples()
        # Train a candidate new model on all examples
        candidate_model = model.fit_new_model(examples)
        # Let the model and its candidate compete for a number of games
        wins = Pit(candidate_model,
                   model,
                   game_setup,
                   num_duels=num_duel,
                   num_sims=num_sims_duel).play()
        frac_win = wins[0] / sum(wins) if sum(wins) != 0 else 0.5
        # If the candidate model won more than the set threshold, reject the old model
        if frac_win > win_threshold:
            model = candidate_model
            model_counter += 1

        # Write to log file
        if iter_log:
            with open(iter_log, 'a' if i else 'w') as f:
                f.write('\nIteration {}\nDuel result: {}\nFrac won by challenger: {}\nModels replaced: {}\n'.format(
                    i, wins, frac_win, model_counter
                ))

    return model


if __name__ == '__main__':
    from v2.games.connect4 import Connect4
    from v2.games.connect4_model import Connect4Model

    _m = Connect4Model()

    _m = policy_iter_self_play(Connect4,
                               _m,
                               num_iter=NUM_ITER,
                               # num_epis=NUM_EPIS,
                               num_epis=5,
                               num_exam=NUM_EXAM,
                               # num_sims_epis=NUM_SIMS_EPIS,
                               num_sims_epis=20,
                               # num_sims_duel=NUM_SIMS_DUEL,
                               num_sims_duel=20,
                               # num_duel=NUM_DUEL,
                               num_duel=10,
                               num_expl=NUM_EXPL,
                               win_threshold=WIN_THRESHOLD,
                               log_dir='testlog')
