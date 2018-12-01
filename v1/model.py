import copy

from v1.game import GameState


class Model:
    """
    Model wrapper class
    """
    def predict(self, s: GameState):
        """
        Compute a policy and value (between -1 and +1) for this game state in which
            - The policy is a probability vector over all valid actions from s
            - The value is a prediction of who will win the game (+1 for win, -1 for loss)
        :param s: The game state
        :return: A two-tuple of (policy, value)
        """
        raise Exception("Unimplemented!")

    def fit_new_model(self, examples: list):
        """
        Return a trained copy of this model
        :param examples: The examples which are used to train the copy
        :return: A trained deep copy of this model
        """
        raise Exception("Unimplemented!")


class DummyModel(Model):
    """
        Dummy model for testing purposes
    """
    def predict(self, s: GameState):
        actions = s.get_possible_moves()
        return {a: 1 / len(actions) for a in actions}, 0.5

    def fit_new_model(self, examples: list):
        return copy.deepcopy(self)
