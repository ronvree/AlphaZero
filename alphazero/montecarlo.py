import collections
import copy
import argparse
from math import sqrt

import numpy as np

from alphazero.game import GameState
from alphazero.model import Model
from alphazero.util import Distribution


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

    def __init__(self, args: argparse.Namespace):
        """
        Create a new Monte Carlo Search Tree
        :param args: Parsed arguments containing hyperparameters
                        - c_puct: parameter determining the level of exploration during searches
                        - epsilon: parameter controlling the influence of the Dirichlet noise added to the root node
                                   policy
                        - alpha: parameterizes the Dirichlet distribution used to generate additional noise for the
                                 policy at the root of the search tree
        """
        assert args.c_puct >= 0
        assert 0 <= args.epsilon <= 1
        assert 0 <= args.alpha <= 1

        self.c_puct = args.c_puct
        self.epsilon = args.epsilon
        self.alpha = args.alpha
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

    def _search(self, state: GameState, model: Model, root: bool = False):
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
            noise = np.random.dirichlet([self.alpha] * len(p))
            for i, (a, pa) in enumerate(p.items()):
                p[a] = (1 - self.epsilon) * p[a] + self.epsilon * noise[i]

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

    def pi(self, state: GameState, temperature: float = 1):
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
            max_action = max(n.keys(), key=n.get)
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

    def action(self, state: GameState, temperature: float = 1):
        """
        Sample an action from the policy obtained from the search tree
        :param state: The state from which the action should be taken
        :param temperature: A scalar influencing degree of exploration
        :return: A two-tuple of (the sampled action, the policy it was sampled from)
        """
        assert 0 <= temperature <= 1
        # Obtain the distribution
        pi = self.pi(state, temperature)
        return Distribution(pi).sample(), pi


if __name__ == '__main__':
    from alphazero.games.tictactoe.tictactoe import TicTacToe
    from alphazero.model import DummyModel

    _args = argparse.Namespace()
    _args.c_puct = 1
    _args.epsilon = 0.25
    _args.alpha = 0.03

    _tree = MCST(_args)

    _s = TicTacToe(_args)
    _m = DummyModel(TicTacToe, _args)

    while not _s.is_terminal():
        # _tree.clear()
        for _ in range(100):
            _tree.search(_s, _m)
        _a, _pi = _tree.action(_s, temperature=0.5)
        _s.do_move(_a)

    print(_s)
