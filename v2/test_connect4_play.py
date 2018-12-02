import numpy as np

from v2.connect4 import Connect4
from v2.connect4_model import Connect4Model
from v2.self_play_simple import sample_action_from_model


def play_model(start=True):

    model = Connect4Model()
    model.model.load_weights('checkpoint.pth.tar')

    state = Connect4()

    while not state.is_terminal():

        if start:
            print("Current game state:")
            print(state)
            print("Choose from possible actions: (by index)")
            actions = state.get_actions()
            print(list(enumerate(actions)))
            input_index = int(input())
            a = actions[input_index]
        else:
            from v2.self_play_simple import sample_action_from_model
            a = sample_action_from_model(model, state, take_max=True)

        state.do_move(a)
        start = not start
        print(state)

    return model


def play_model_vs_random(rounds=100):
    model = Connect4Model()
    model.model.load_weights('checkpoint.pth.tar')

    wins = [0, 0]

    for r in range(rounds):
        state = Connect4()
        start = bool(r % 2)
        while not state.is_terminal():

            if start:
                actions = state.get_actions()
                a = actions[np.random.randint(0, len(actions))]
            else:
                a = sample_action_from_model(model, state, take_max=True)

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


if __name__ == '__main__':

    # _m = play_model()
    _m = play_model_vs_random()

