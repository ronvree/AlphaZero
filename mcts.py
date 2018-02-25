import numpy as np
from math import sqrt

from game import GameState
from model import Model


class MonteCarloSearchTree(dict):

    def __init__(self, c_puct=1, **kwargs):
        super().__init__(**kwargs)
        self.c_puct = c_puct

    def search(self, state: GameState, model: Model):
        if state.is_terminal():  # If the state is terminal, return the score  TODO -- negate???
            return state.get_score()

        actions = state.get_possible_moves()
        if state not in self.keys():  # Expand the search tree
            # Store for each node:
            # - Q: The expected reward for taking action a from the game state
            # - N: The number of times action a was performed from the game state during simulations
            # - P: A policy determining which move to take as decided by the neural network
            # - v: A value for this game state (from the current player's perspective) as decided by the neural network
            q, n = {a: 0 for a in actions}, {a: 0 for a in actions}
            p, v = model.predict(state)
            self[hash(state)] = (p, v, q, n)
            return v

        # The game state already occurs in the search tree
        p, v, q, n = self[hash(state)]  # Recall policy p, value v, expected reward q, simulation count n

        # Pick an action to explore by maximizing U, the upper confidence bound on the Q-values
        _, a = max([(q[a] + self.c_puct * p[a] * sqrt(sum(n)) / (1 + n[a]), a) for a in actions],
                   key=lambda x: x[0])

        # Perform the selected action on the state and continue the simulation
        # Negate the value returned, as the 'current player' perspective is changed
        v = -self.search(state.do_move(a), model)

        # Propagate the reward back up the tree
        q[a] = (n[a] * q[a] + v) / (n[a] + 1)
        n[a] += 1
        return v

    def pi(self, state: GameState, temperature=1):
        """
        Give an improved policy based on the counts in N
        :param state: The state from which the policy should be obtained
        :param temperature: Scalar influencing degree of exploration
        :return: a dictionary mapping each valid action to a probability
        """
        _, _, _, n = self[hash(state)]

        if temperature == 0:
            # The action with highest count should be chosen with probability=1
            a_max = max(n.keys(), key=n.get)
            return {a: 1 if a == a_max else 0 for a in n.keys()}
        else:
            # The actions should be chosen proportional to N(s, a)^(1 / temperature)
            n_temp = {a: v ** (1 / temperature) for a, v in n.items()}
            # Sum total for normalization
            n_total = sum(n_temp.values())
            if n_total == 0:
                return {a: 1 / len(n) for a in n.keys()}
            else:
                return {a: v / n_total for a, v in n_temp.items()}

    def action(self, state: GameState, temperature=1):
        """
        Sample an action from the policy obtained from the search tree
        :param state: The state from which the action should be taken
        :param temperature: A scalar influencing degree of exploration
        :return: A two-tuple of (the sampled action, the policy it was sampled from)
        """
        pi = self.pi(state, temperature)
        # Introduce an ordering in the items
        items = list(pi.items())
        # So an random.choice can select an index with probability p for picking an action
        return items[np.random.choice(len(items), p=[p[1] for p in items])][0], pi


if __name__ == '__main__':
    from games.connect4 import Connect4
    from model import DummyModel

    tree = MonteCarloSearchTree()

    s = Connect4()
    m = DummyModel()

    tree.search(s, m)
    tree.search(s, m)

    print(tree)
