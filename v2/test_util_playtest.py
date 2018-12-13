import numpy as np

from v2.game import GameState
from v2.model import Model
from v2.montecarlo import MCST


def play(game_init: callable, p1: callable, p2: callable):
    state = game_init()
    current = True
    while not state.is_terminal():
        a = p1(state) if current else p2(state)
        state.do_move(a)
        current = not current
        print(state)
    return state.get_reward()


def play_n(game_init: callable, agent1: callable, agent2: callable, n=100):
    wins = [0, 0]
    for r in range(n):
        start = bool(r % 2)
        if start:
            r_win = play(game_init, agent1, agent2)
        else:
            r_win = play(game_init, agent2, agent1)

        if r_win == 1:
            if start:
                wins[1] += 1
            else:
                wins[0] += 1
        if r_win == -1:
            if start:
                wins[0] += 1
            else:
                wins[1] += 1
    return wins


def play_manual(game_init: callable, agent: callable, start=True):
    if start:
        return play(game_init, ask_move, agent)
    else:
        return play(game_init, agent, ask_move)


def play_random(game_init: callable, agent: callable, start=True):
    if start:
        return play(game_init, agent, random_move)
    else:
        return play(game_init, random_move, agent)


def ask_move(state: GameState):
    print("Current game state:")
    print(state)
    print("Choose from possible actions: (by index)")
    actions = state.get_actions()
    print(list(enumerate(actions)))
    input_index = int(input())
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
    from v2.connect4 import Connect4
    from v2.connect4_model import Connect4Model

    _m = Connect4Model()
    _m.model.load_weights('checkpoint.pth.tar')

    _t = MCST()

    play_manual(Connect4, lambda s: ask_model_mcst(s, _m, _t, temperature=0, searches=1000))
