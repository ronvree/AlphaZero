import numpy as np
import torch
import torch.cuda
import argparse

from v3.game import GameState
from v3.games.connect4.connect4 import Connect4
# from v3.model_keras import ResidualNeuralNetwork as KerasResNet
from v3.model_pytorch import ResidualNetwork as TorchResNet


# class Connect4Model(KerasResNet):
#
#     def __init__(self, args: argparse.Namespace):
#         super().__init__(Connect4,
#                          args,
#                          num_resd_block=5,
#                          conv_block_params={'kernel_size': (4, 4)},
#                          resd_block_params={'kernel_size': (4, 4)})
#
#     def state_input(self, s: Connect4):
#         return np.reshape(s.grid * s.current_player, newshape=self.input_shape)
#
#     def deepcopy(self):
#         self.model.save_weights('checkpoint.pth.tar')
#         self_copy = Connect4Model(self.args)
#         self_copy.model.load_weights('checkpoint.pth.tar')
#         return self_copy

class Connect4Model(TorchResNet):

    def __init__(self, args: argparse.Namespace):
        args.input_channels = 1
        args.num_res_blocks = 5
        args.block_channels = 32
        args.conv_block_kernel_size = (5, 5)
        args.conv_block_padding = 2
        args.res_block_kernel_size = (5, 5)
        args.res_block_padding = 2
        super().__init__(Connect4, args)

    def state_input(self, s: GameState):
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        return torch.Tensor(s.grid * s.get_current_player()).unsqueeze(0).to(device)

    def deepcopy(self):
        """
        Create a deep copy of this model
        :return: a deep copy of this model
        """
        model_copy = type(self)(self.args)
        model_copy.load_state_dict(self.state_dict())
        model_copy.get_optimizer().load_state_dict(self.get_optimizer().state_dict())
        return model_copy
