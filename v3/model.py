import argparse

from v3.game import GameState


class Model:
    """
    Model wrapper class
    """

    def __init__(self, game: callable, args: argparse.Namespace):
        """
        Create a new Model
        :param game: The game that the model should play
        :param args: Parsed arguments containing hyperparameters of the model
        """
        self.game = game
        self.args = args

    def predict(self, s: GameState) -> tuple:
        """
        Compute a policy and value (between -1 and +1) for this game state in which
            - The policy is a probability vector over all valid actions from s
            - The value is a prediction of who will win the game (+1 for win, -1 for loss)
        :param s: The game state
        :return: A two-tuple of (policy, value)
        """
        raise NotImplementedError

    def fit(self, examples: list):
        """
        Train the model on the given examples
        :param examples: The examples on which the model should be trained
        """
        raise NotImplementedError

    def deepcopy(self):
        """
        :return: A deep copy of this model
        """
        raise NotImplementedError

    def save(self, directory_path: str, examples: list = None):
        """
        Save the model to a specific directory
        :param directory_path: The path to the directory to which the model should be saved
        :param examples: Optional list of training examples that can be saved with the model
        """
        raise NotImplementedError

    @staticmethod
    def load(directory_path: str, load_examples: bool = False) -> tuple:
        """
        Load a model from the specified directory
        :param directory_path: The directory from which the model should be loaded
        :param load_examples: Indicates whether examples should be loaded with the model if present
        :return: A two-tuple of:
                    - The loaded model
                    - Optional list of training examples that were saved with the model
        """
        raise NotImplementedError


class DummyModel(Model):
    """
        Dummy model for testing purposes
    """

    def __init__(self, game: callable, args: argparse.Namespace):
        super().__init__(game, args)

    def predict(self, s: GameState) -> tuple:
        actions = s.get_actions()
        return {a: 1 / len(actions) for a in actions}, 0.5

    def fit(self, examples: list):
        pass  # Do nothing

    def deepcopy(self):
        return DummyModel(self.game, self.args)

    def save(self, directory_path: str, examples: list = None):
        pass

    @staticmethod
    def load(directory_path: str):
        pass
