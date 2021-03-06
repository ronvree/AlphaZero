import argparse
from collections import deque

from tqdm import tqdm

from alphazero.duel import Pit
from alphazero.episode import Episode
from alphazero.model import Model

"""
    Train the model through policy iteration

"""


def policy_iter_self_play(game: callable,
                          model: Model,
                          args: argparse.Namespace,
                          examples: list = None,
                          ):
    """
    Train the model through self-play policy iteration
    :param game: The constructor of the game that should be played
    :param model: The model that should be trained
    :param args: Parsed arguments containing hyperparameters
    :param examples: Train examples from previous self plays that can be used during training
    :return: the trained model
    """
    examples = examples or []
    examples = deque(examples, maxlen=args.num_exam)
    model_counter = 0  # Count number of models that have been replaced so far
    for i in range(args.num_iter):
        # Play a number of games of self-play and obtain examples to learn from
        # Each example is a tuple of (game state, corresponding policy, final reward)
        for _ in tqdm(range(args.num_epis)):
            examples += Episode(game,
                                model,
                                args).get_examples()
        # Train a candidate new model on all examples
        candidate_model = fit_new_model(model, examples)
        # Let the model and its candidate compete for a number of games
        wins = Pit(candidate_model,
                   model,
                   game,
                   args).play()
        frac_win = wins[0] / sum(wins) if sum(wins) != 0 else 0.5
        # If the candidate model won more than the set threshold, reject the old model
        if frac_win > args.win_threshold:
            model = candidate_model
            model_counter += 1
        # Occasionally save the model
        if i % args.save_interval == 0:
            model.save('./checkpoint', examples=examples)
    return model


def fit_new_model(model: Model, examples: list):
    """
    Train a copy of the model on the given examples
    :param model: The model that should be copied
    :param examples: The examples on which the copy should be trained
    :return: the trained copied model
    """
    candidate = model.deepcopy()
    candidate.fit(examples)
    return candidate


if __name__ == '__main__':
    from alphazero.games.tictactoe.tictactoe import TicTacToe
    from alphazero.model import DummyModel

    _args = argparse.Namespace()
    _args.num_iter = 10000
    _args.num_epis = 5
    _args.num_exam = 5000
    _args.num_sims_epis = 20
    _args.num_sims_duel = 20
    _args.num_duel = 10
    _args.num_expl = 30
    _args.win_threshold = 0.55
    _args.c_puct = 1
    _args.epsilon = 0.25
    _args.alpha = 0.03

    _m = DummyModel(TicTacToe, _args)

    _m = policy_iter_self_play(TicTacToe,
                               _m,
                               _args)
