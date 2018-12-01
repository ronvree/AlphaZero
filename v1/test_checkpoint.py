from v1.connect4_model import TestNetwork
from v1.game import GameState
from v1.games.connect4 import Connect4
from v1.mcts import MonteCarloSearchTree
from v1.model import Model


def play(game: callable, p1: callable, p2: callable):
    players = [p1, p2]
    state = game()
    current = 0
    while not state.is_terminal():
        state.do_move(players[current](state))
        current = 1 - current
    print(state)  # TODO -- netjes
    return state.get_scores()


def play_model(game: callable, p: callable, model: Model, start: bool=True, nr_of_mcts_simulations: int=100):
    mcts = MonteCarloSearchTree()

    def determine_move(state):
        for _ in range(nr_of_mcts_simulations):
            mcts.search(state, model)
        return mcts.action(state, temperature=0)[0]

    if start:
        return play(game, p, determine_move)
    else:
        return play(game, determine_move, p)


def play_random(game: callable, model: Model, n: int=100, nr_of_mcts_simulations: int=100, verbose: bool=True):
    win_counts = np.zeros(2)
    for i in range(n):
        if verbose:
            print('Playing game {}'.format(i))
        score = play_model(game, random_input, model, start=bool(i % 2), nr_of_mcts_simulations=nr_of_mcts_simulations)
        if i % 2 == 0:
            win_counts += score
        else:
            win_counts += np.roll(score, 1)
    # return win_counts[0] / sum(win_counts)
    return win_counts


def human_input(state: GameState):
    print("Current game state:")
    print(state)
    print("Choose from possible actions: (by index)")
    actions = state.get_possible_moves()
    print(list(enumerate(actions)))
    input_index = int(input())
    return actions[input_index]


def random_input(state: GameState):
    actions = state.get_possible_moves()
    return actions[np.random.randint(len(actions))]


if __name__ == '__main__':
    import numpy as np

    checkpoint_model = TestNetwork()
    checkpoint_model.model.load_weights('checkpoint.pth.tar')

    # play_model(Connect4, human_input, checkpoint_model, start=False, nr_of_mcts_simulations=40)

    # play(Connect4, human_input, human_input)

    win_frac = play_random(Connect4, checkpoint_model, n=700, nr_of_mcts_simulations=40)
    print(win_frac)
