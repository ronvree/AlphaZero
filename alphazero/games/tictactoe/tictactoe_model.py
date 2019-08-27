import numpy as np
import torch

import argparse
import pickle
import os

from alphazero.game import GameState
from alphazero.model_keras import ResidualNeuralNetwork as KerasResNet
from alphazero.model_pytorch import ResidualNetwork as TorchResNet
from alphazero.games.tictactoe.tictactoe import TicTacToe


# class TicTacToeModel(KerasResNet):
#
#     def __init__(self, args: argparse.Namespace):
#         super().__init__(TicTacToe,
#                          args,
#                          num_resd_block=3,
#                          conv_block_params={'filters': 32},
#                          resd_block_params={'filters': 32})
#
#     def state_input(self, s: TicTacToe):
#         return np.reshape(s.grid * s.current_player, newshape=self.input_shape)
#
#     def deepcopy(self):
#         self.model.save_weights('checkpoint.pth.tar')
#         self_copy = TicTacToeModel(self.args)
#         self_copy.model.load_weights('checkpoint.pth.tar')
#         return self_copy


class TicTacToeModel(TorchResNet):

    def __init__(self, args: argparse.Namespace):
        args.input_channels = 1
        args.num_res_blocks = 3
        args.block_channels = 32
        args.conv_block_kernel_size = (3, 3)
        args.conv_block_padding = 1
        args.res_block_kernel_size = (3, 3)
        args.res_block_padding = 1
        super().__init__(TicTacToe, args)

    def state_input(self, s: GameState):
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        return torch.Tensor(s.grid * s.get_current_player()).unsqueeze(0).to(device)


if __name__ == '__main__':
    from v2.montecarlo import MCST

    _args = argparse.Namespace()
    _args.c_l2reg = 0.0001
    _args.learning_rate = 0.01
    _args.momentum = 0.9
    _args.disable_cuda = True

    _s = TicTacToe(_args).do_move((0, 0))

    _m = TicTacToeModel(_args)

    _p, _v = _m.predict(_s)

    print(_p, _v)

    _t = MCST()

    for _ in range(2000):
        _t.search(_s, _m)

    _a, _pi = _t.action(_s, temperature=1)

    print(_a, _pi)

