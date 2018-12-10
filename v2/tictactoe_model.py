import numpy as np

from v2.model import ResidualNeuralNetwork
from v2.tictactoe import TicTacToe


class TicTacToeModel(ResidualNeuralNetwork):

    def __init__(self):
        super().__init__(TicTacToe,
                         input_shape=TicTacToe.board_shape(),
                         num_resd_block=3,
                         conv_block_params={'filters': 32},
                         resd_block_params={'filters': 32})

    def state_input(self, s: TicTacToe):
        return np.reshape(s.get_observation(), newshape=self.input_shape)

    def deepcopy(self):
        self.model.save_weights('checkpoint.pth.tar')
        self_copy = TicTacToeModel()
        self_copy.model.load_weights('checkpoint.pth.tar')
        return self_copy


if __name__ == '__main__':
    from v2.montecarlo import MCST

    _s = TicTacToe().do_move((0, 0))

    _m = TicTacToeModel()

    _p, _v = _m.predict(_s)

    print(_p, _v)

    _t = MCST()

    for _ in range(20):
        _t.search(_s, _m)

    _a, _pi = _t.action(_s, temperature=1)

    print(_a, _pi)

