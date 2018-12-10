import collections
import copy
from math import sqrt

import numpy as np

from v2.game import GameState
from v2.model import Model


class MCST(collections.MutableMapping):
    """
        Monte Carlo Search Tree class

        Game states form nodes in this search tree. Available actions represent outgoing edges for each node.
        Game states are stored as hashes, so information about the states cannot be retrieved easily.

        Stores for each encountered state the following:
         - Q: The expected reward for taking action a from the game state
         - N: The number of times action a was performed from the game state during simulations
         - P: A policy determining which move to take as decided by the neural network
         - v: A value for this game state (from the current player's perspective) as decided by the neural network
    """

    def __init__(self, c_puct: float=1.0, epsilon: float=0.25, alpha: float=0.03):
        """
        Create a new Monte Carlo Search Tree
        :param c_puct: Constant determining the level of exploration in action selection during the search procedure
        :param alpha: Parameter of the Dirichlet distribution that is used to add noise to action selection
                      in root nodes
        :param epsilon: Constant determining the influence of noise in action selection in the root nodes
        """
        assert c_puct >= 0
        assert 0 <= epsilon <= 1
        assert 0 <= alpha <= 1
        self.c_puct = c_puct
        self.epsilon = epsilon
        self.alpha = alpha
        self.store = dict()

    @staticmethod
    def key_transform(key):
        """
        Function applied on the key before storing it in the search tree
        :param key: Key (game state) to be stored in the search tree
        :return: The hash value of the key/game state
        """
        return hash(key)

    def __getitem__(self, k: GameState):
        return self.store[MCST.key_transform(k)]

    def __setitem__(self, k: GameState, v: tuple):
        self.store[MCST.key_transform(k)] = v

    def __delitem__(self, k: GameState):
        del self.store[MCST.key_transform(k)]

    def __len__(self) -> int:
        return len(self.store)

    def __iter__(self):
        for h in self.store.keys():
            yield h

    def __contains__(self, item: GameState):
        return MCST.key_transform(item) in self.store.keys()

    def clear(self):
        """
        :return: Removes all nodes/edges/attributes in the search tree
        """
        self.store.clear()

    def search(self, state: GameState, model: Model):
        """
        Perform Monte Carlo Tree Search on a deep copy of the state
        :param state: The game state on which the mcts should be performed
        :param model: The model that will provide the mcts with (policy, value) estimations
        :return: The z-value, corresponding to whether the search resulted in a win (z=1), loss (z=-1) or draw (z=0)
        """
        return self._search(copy.deepcopy(state), model, root=True)

    def _search(self, state: GameState, model: Model, root: bool=False):
        """
        Helper function for self.search allowing for recursive search calls using only one state instance
        :param state: The game state on which the mcts should be performed
        :param model: The model that will provide the mcts with (policy, value) estimations
        :param root: A boolean indicating whether this is the root of the search tree. Dirichlet noise is added to the
                     policy at the root to increase exploration
        :return: The z-value, corresponding to whether the search resulted in a win (z=1), loss (z=-1) or draw (z=0)
                 The z-value is negated once returned to account for changing player perspectives
        """
        if state.is_terminal():
            return -state.get_reward()

        actions = state.get_actions()
        if state not in self:  # Expand the search tree
            # Store for each node:
            # - Q: The expected reward for taking action a from the game state
            # - N: The number of times action a was performed from the game state during simulations
            # - P: A policy determining which move to take as decided by the neural network
            # - v: A value for this game state (from the current player's perspective) as decided by the neural network
            q, n = {a: 0 for a in actions}, {a: 0 for a in actions}
            p, v = model.predict(state)
            self[state] = (p, v, q, n)
            return -v

        # The game state already occurs in the search tree
        p, v, q, n = self[state]  # Recall policy p, value v, expected reward q, simulation count n

        # Add noise to the action selection in the root node for increased exploration
        if root:
            for a, pa in p.items():
                p[a] = (1 - self.epsilon) * p[a] + self.epsilon * np.random.dirichlet([self.alpha])

        # Pick an action to explore by maximizing U, the upper confidence bound on the Q-values
        u = {a: self.c_puct * p[a] * sqrt(sum(n.values())) / (1 + n[a]) for a in actions}
        a = max(actions, key=lambda a: q[a] + u[a])

        # Perform the selected action on the state and continue the simulation
        # Negate the value returned, as the 'current player' perspective is changed
        v = self._search(state.do_move(a), model)

        # Propagate the reward back up the tree
        q[a] = (n[a] * q[a] + v) / (n[a] + 1)
        n[a] += 1
        return -v

    def pi(self, state: GameState, temperature: float=1):
        """
        Give an improved policy based on the counts in N
        :param state: The state from which the policy should be obtained
        :param temperature: Scalar influencing degree of exploration
                            A higher temperature gives a larger focus on the move with highest N
        :return: a dictionary mapping each valid action to a probability
        """
        assert 0 <= temperature <= 1

        _, _, _, n = self[state]

        if temperature == 0:
            # The action with highest count should be chosen with probability=1
            # max_actions = [a for a in n.keys() if n[a] == max(n)]
            max_action = max(n.keys(), key=n.get)  # TODO -- pick randomly between equal highest n?
            return {action: 1 if action == max_action else 0 for action in n.keys()}
        else:
            # The actions should be chosen proportional to N(s, a)^(1 / temperature)
            n_temp = {action: count ** (1 / temperature) for action, count in n.items()}
            # Sum total for normalization
            n_total = sum(n_temp.values())
            if n_total == 0:
                return {action: 1 / len(n) for action in n.keys()}
            else:
                return {action: count / n_total for action, count in n_temp.items()}

    def action(self, state: GameState, temperature: float=1):
        """
        Sample an action from the policy obtained from the search tree
        :param state: The state from which the action should be taken
        :param temperature: A scalar influencing degree of exploration
        :return: A two-tuple of (the sampled action, the policy it was sampled from)
        """
        assert 0 <= temperature <= 1

        pi = self.pi(state, temperature)
        # Introduce an ordering in the (action, probability) pairs
        items = list(pi.items())
        # So a random.choice can select an index with probability p for picking an action
        return items[np.random.choice(len(items), p=[p[1] for p in items])][0], pi


if __name__ == '__main__':
    from v2.connect4 import Connect4
    from v2.model import DummyModel

    _tree = MCST(c_puct=1)

    _s = Connect4()
    _m = DummyModel()

    while not _s.is_terminal():
        # _tree.clear()
        for _ in range(100):
            _tree.search(_s, _m)
        _a, _pi = _tree.action(_s, temperature=0.5)
        _s.do_move(_a)

    print(_s)
