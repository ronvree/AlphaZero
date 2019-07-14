
import numpy as np

from v3.game import GameState
from v3.games.connect4.connect4 import Connect4
from v3.games.connect4.connect4_model import Connect4Model
from v3.tests.test_util_args import get_dummy_args
from v3.util import Distribution


def play_model(start=True):

    args = get_dummy_args()

    model = Connect4Model(args)
    model.model.load_weights('./saved_model/weights.h5')

    state = Connect4(args)

    while not state.is_terminal():

        if start:
            a = ask_move(state)
        else:
            ps, v = model.predict(state)
            a = Distribution(ps).sample_max()

        state.do_move(a)
        start = not start
        print(state)

    return model


def play_model_vs_random(rounds=100):
    args = get_dummy_args()

    model = Connect4Model(args)
    model.model.load_weights('./saved_model/weights.h5')

    wins = [0, 0]

    for r in range(rounds):
        state = Connect4(args)
        start = bool(r % 2)
        while not state.is_terminal():

            if start:
                actions = state.get_actions()
                a = actions[np.random.randint(0, len(actions))]
            else:
                ps, v = model.predict(state)
                a = Distribution(ps).sample_max()

            state.do_move(a)
            start = not start

        if state.get_reward() == 1:
            if start:
                wins[1] += 1
            else:
                wins[0] += 1
        if state.get_reward() == -1:
            if start:
                wins[0] += 1
            else:
                wins[1] += 1

        print(state)
    print(wins)
    return model


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


if __name__ == '__main__':

    _m = play_model()

