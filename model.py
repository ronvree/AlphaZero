import copy
import numpy as np

from game import GameState


class Model:

    def predict(self, s: GameState):
        """
        Compute a score and policy for this game state
        :param s: The game state
        :return: A two-tuple of (policy, value)
        """
        raise Exception("Unimplemented!")

    def fit_new_model(self, examples: list):
        raise Exception("Unimplemented!")

    # def predict_action(self, s: GameState):
    #     policy, _ = self.predict(s)
    #     actions = s.get_possible_moves()
    #     # return max(actions, key=policy.get)
    #     return actions[np.random.choice(len(actions), p=list(policy.values()))]


class DummyModel(Model):

    def predict(self, s: GameState):
        actions = s.get_possible_moves()
        return {a: 1 / len(actions) for a in actions}, 1

    def fit_new_model(self, examples: list):
        return copy.deepcopy(self)

