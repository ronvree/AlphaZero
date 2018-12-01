
from v2.connect4 import Connect4
from v2.model import ResidualNeuralNetwork


class Connect4Model(ResidualNeuralNetwork):

    def __init__(self):
        super().__init__(Connect4,
                         num_resd_block=5)

    def deepcopy(self):
        self.model.save_weights('checkpoint.pth.tar')
        self_copy = Connect4Model()
        self_copy.model.load_weights('checkpoint.pth.tar')
        return self_copy
