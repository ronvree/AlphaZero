
import numpy as np

from v3.game import GameState
from v3.games.connect4.connect4 import Connect4
from v3.games.connect4.connect4_model import Connect4Model
from v3.montecarlo import MCST
from v3.tests.test_util_args import get_dummy_args
from v3.util import Distribution


def play_model(start=True):

    args = get_dummy_args()

    model, _ = Connect4Model.load('./checkpoint')

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


def play_model_mcst(start=True, mcst_searches=100, temperature=0):
    model, _ = Connect4Model.load('./checkpoint')
    state = Connect4(model.args)
    mcst = MCST(model.args)
    while not state.is_terminal():
        if start:
            a = ask_move(state)
        else:
            for _ in range(mcst_searches):
                mcst.search(state, model)
            a, _ = mcst.action(state, temperature)

        state.do_move(a)
        start = not start
        print(state)
    return model, mcst


def play_model_vs_random(rounds=100):
    args = get_dummy_args()

    model, _ = Connect4Model.load('./checkpoint')

    wins = [0, 0]

    for r in range(rounds):
        state = Connect4(args)
        turn = bool(r % 2)
        while not state.is_terminal():

            if turn:
                actions = state.get_actions()
                a = actions[np.random.randint(0, len(actions))]
            else:
                ps, v = model.predict(state)
                a = Distribution(ps).sample_max()

            state.do_move(a)
            turn = not turn

        if state.get_reward() == 1:
            if turn:
                wins[1] += 1
            else:
                wins[0] += 1
        if state.get_reward() == -1:
            if turn:
                wins[0] += 1
            else:
                wins[1] += 1

        print(state)
    print(wins)
    return model


def play_model_mcst_vs_random(rounds=100, mcst_searches=100, temperature=0):
    model, _ = Connect4Model.load('./checkpoint')

    wins = [0, 0]

    for r in range(rounds):
        state = Connect4(model.args)
        mcst = MCST(model.args)
        turn = bool(r % 2)
        while not state.is_terminal():

            if turn:
                actions = state.get_actions()
                a = actions[np.random.randint(0, len(actions))]
            else:
                for _ in range(mcst_searches):
                    mcst.search(state, model)

                a, _ = mcst.action(state, temperature)

            state.do_move(a)
            turn = not turn

        if state.get_reward() == 1:
            if turn:
                wins[1] += 1
            else:
                wins[0] += 1
        if state.get_reward() == -1:
            if turn:
                wins[0] += 1
            else:
                wins[1] += 1

        print(state)
    print(wins)
    return model, mcst


def play_model_mcst_vs_random_mcst(rounds=100, mcst_searches=100, temperature=0):
    model, _ = Connect4Model.load('./checkpoint')
    opponent = Connect4Model(model.args)
    wins = [0, 0]
    for r in range(rounds):
        state = Connect4(model.args)
        mcst = MCST(model.args)
        mcst_opponent = MCST(model.args)
        turn = bool(r % 2)
        while not state.is_terminal():
            if turn:
                for _ in range(mcst_searches):
                    mcst_opponent.search(state, opponent)
                a, _ = mcst_opponent.action(state, temperature)
            else:
                for _ in range(mcst_searches):
                    mcst.search(state, model)
                a, _ = mcst.action(state, temperature)
            state.do_move(a)
            turn = not turn

        if state.get_reward() == 1:
            if turn:
                wins[1] += 1
            else:
                wins[0] += 1
        if state.get_reward() == -1:
            if turn:
                wins[0] += 1
            else:
                wins[1] += 1

        print(state)
    print(wins)
    return model, mcst


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

    # _m = play_model()
    # _m = play_model_vs_random()

    # play_model_mcst(mcst_searches=1000)
    # play_model_mcst_vs_random()

    play_model_mcst_vs_random_mcst()

