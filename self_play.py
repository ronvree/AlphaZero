import numpy as np
from tqdm import tqdm

from mcts import MonteCarloSearchTree
from model import Model

number_of_iterations = 1000000
number_of_episodes = 100
number_of_mcts_simulations = 40
number_of_duels = 30
number_of_exploring_moves = 30
threshold = 0.55


def policy_iter_self_play(game: callable, model_type: callable):
    # Initialize a random model
    model = model_type()
    examples = []
    model_counter = 0  # TODO -- remove
    iter_counter = 0  # TODO -- remove
    for i in range(number_of_iterations):
        # Play a number of games of self-play and obtain examples to learn from
        # Each example is a tuple of (game state, corresponding policy, final reward)
        for _ in tqdm(range(number_of_episodes)):
            examples += execute_episode(game, model)
        # Train a candidate new model on all examples
        candidate_model = model.fit_new_model(examples)
        # Let the model and its candidate compete for a number of games
        frac_win = pit(candidate_model, model, game)
        # If the candidate model won more than the set threshold, reject the old model
        if frac_win > threshold:
            model = candidate_model
            model_counter += 1
        print('Fraction of games won: {}'.format(frac_win))
        print('Models replaced so far: {}'.format(model_counter))
        print('Iteration {} complete'.format(iter_counter))
        iter_counter += 1
    return model


def execute_episode(game: callable, model: Model):
    examples = []
    mcts = MonteCarloSearchTree()
    # Initialize the game state
    state = game()
    # Continue until the game ends
    move_counter = 0
    while not state.is_terminal():
        # Perform a number of Monte Carlo Tree Search simulations
        for _ in range(number_of_mcts_simulations):
            mcts.search(state, model)

        # The first moves have a temperature of 0 for increased exploration
        tau = 0 if move_counter < number_of_exploring_moves else 1

        # Obtain an improved policy from the tree
        a, pi = mcts.action(state, temperature=tau)

        # Store examples for the model to train on. One example consists of:
        # - The game state from the current player's perspective
        # - A policy obtained from the search tree
        # - A value indicating if the player eventually won the game (so is added later)
        examples.append([state.get_observation(), pi])

        # Perform the move
        state = state.do_move(a)
        move_counter += 1

    # If the game ended, add the final result to the examples and return them
    # reward=1 if player 1 won, reward=-1 if player 2 won
    reward = state.get_score()
    for e in examples:
        e.append(reward)
        reward *= -1  # Switch player perspective
    return examples


def pit(m_alt: Model, m_orig: Model, game: callable):
    # Keep track of the number of games won for each model
    wins = np.zeros(2)

    # Let the models play a number of games
    for duel in range(number_of_duels):
        # Initialize a search tree for both models
        mcts_alt, mcts_orig = MonteCarloSearchTree(), MonteCarloSearchTree()

        # Store which model corresponds to which player
        # Let the models take turns in who is the starting player
        models = {duel % 2: (m_alt, mcts_alt), (duel + 1) % 2: (m_orig, mcts_orig)}

        # Initialize a new game and play it
        state = game()
        current_player = 0
        while not state.is_terminal():
            model, tree = models[current_player]
            for _ in range(number_of_mcts_simulations):
                tree.search(state, model)
            # Determine an action by sampling from the policy as defined by the tree
            a, _ = tree.action(state, temperature=1)  # Temperature = 1 for best move selection (no exploration)
            state.do_move(a)
            current_player = 1 - current_player
        # Add the game result to the win counter (taking player perspective into account)
        if duel % 2 == 0:
            wins += state.get_scores()
        else:
            wins += np.roll(state.get_scores(), 1)
    # Return the fraction of games won by the alternative model
    if sum(wins) == 0:
        return 0.5
    else:
        return wins[0] / sum(wins)


if __name__ == '__main__':
    from connect4_model import TestNetwork
    from games.connect4 import Connect4
    from games.ringgz import Ringgz
    from model import DummyModel
    import ringgz2_model
    from games.ringgz_2 import Ringgz2

    # s = Connect4()
    # m = DummyModel()

    # exs = execute_episode(s, m)

    # m = policy_iter_self_play(Connect4, TestNetwork)
    m = policy_iter_self_play(Ringgz2, ringgz2_model.TestNetwork)
    # m = policy_iter_self_play(lambda: Ringgz(2), DummyModel)

    print(m)

    pass
