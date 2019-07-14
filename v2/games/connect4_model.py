import numpy as np

from v2.games.connect4 import Connect4
from v2.model import ResidualNeuralNetwork


class Connect4Model(ResidualNeuralNetwork):

    def __init__(self):
        super().__init__(Connect4,
                         input_shape=Connect4.board_shape(),
                         num_resd_block=5,
                         conv_block_params={'kernel_size': (4, 4)},
                         resd_block_params={'kernel_size': (4, 4)})

    def state_input(self, s: Connect4):
        return np.reshape(s.grid * s.current_player, newshape=self.input_shape)

    def deepcopy(self):
        self.model.save_weights('checkpoint.pth.tar')
        self_copy = Connect4Model()
        self_copy.model.load_weights('checkpoint.pth.tar')
        return self_copy
