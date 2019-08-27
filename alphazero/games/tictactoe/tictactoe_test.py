import numpy as np
import argparse

from alphazero.game import GameState
from alphazero.model import Model
from alphazero.montecarlo import MCST


def play(game_init: callable, p1: callable, p2: callable, args: argparse.Namespace, show=True):
    state = game_init(args)
    current = True
    while not state.is_terminal():
        a = p1(state) if current else p2(state)
        state.do_move(a)
        current = not current
        if show:
            print(state)
    return state.get_reward()


def play_manual(game_init: callable, agent: callable, args: argparse.Namespace, start=True):
    if start:
        return play(game_init, ask_move, agent, args)
    else:
        return play(game_init, agent, ask_move, args)


def play_random(game_init: callable, agent: callable, args: argparse.Namespace, start=True):
    if start:
        return play(game_init, agent, random_move, args)
    else:
        return play(game_init, random_move, agent, args)


def ask_move(state: GameState):
    print("Current game state:")
    print(state)
    print("Choose from possible actions: (by index)")
    actions = state.get_actions()
    print(list(enumerate(actions)))
    while True:
        input_index = input()
        try:
            input_index = int(input_index)
        except ValueError:
            continue
        if 0 <= input_index < len(actions):
            break
    a = actions[input_index]
    return a


def random_move(state: GameState):
    actions = state.get_actions()
    return actions[np.random.randint(0, len(actions))]


def ask_model(state: GameState, model: Model, take_max: bool = False):
    p, _ = model.predict(state)

    actions = state.get_actions()

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


def ask_model_mcst(state: GameState, model: Model, mcst: MCST, temperature: float = 0.0, searches: int = 100):
    for _ in range(searches):
        mcst.search(state, model)
    a, _ = mcst.action(state, temperature=temperature)
    return a


if __name__ == '__main__':
    from alphazero.games.tictactoe.tictactoe import TicTacToe
    from alphazero.games.tictactoe.tictactoe_model import TicTacToeModel

    from alphazero.tests.test_util_args import get_dummy_args

    _args = get_dummy_args()

    # _m = TicTacToeModel(_args)
    # _m.model.load_weights('./saved_model/weights.h5')

    _m, _ = TicTacToeModel.load('./checkpoint')

    _t = MCST(_args)

    # play_random(TicTacToe, lambda s: ask_model_mcst(s, _m, _t, temperature=0, searches=1000), _args, start=False)

    play_random(TicTacToe, lambda s: ask_model(s, _m, take_max=True), _args, start=False)

    # play_manual(TicTacToe, lambda s: ask_model_mcst(s, _m, _t, temperature=0, searches=1000), _args)
