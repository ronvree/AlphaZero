
import torch
import torch.nn

import argparse

from v3.game import GameState
from v3.model import Model


class TorchModel(Model, torch.nn.Module):

    def __init__(self, game: callable, args: argparse.Namespace):
        super().__init__()
        self.args = args

        self.in_shape = game.board_shape()
        self.action_space = game.action_space()
        self.out_size = len(self.action_space)

        self.action_index = {a: i for i, a in enumerate(self.action_space)}

    def state_input(self, s: GameState):
        raise NotImplementedError

    def predict(self, s: GameState) -> tuple:
        # Convert GameState object to suitable input
        state = self.state_input(s)
        # Obtain predictions of the network
        p, v = self.forward(state)  # p is a policy over all moves of the game
        # Mask all illegal actions
        legal_actions = s.get_actions()
        mask = torch.zeros(self.out_size)
        for a in legal_actions:
            mask[self.action_index[a]] = 1
        p *= mask
        # Normalize the probability distribution so it sums to 1 again
        p /= torch.sum(p)
        # Return the probability distribution as dict
        p = {a: p[self.action_index[a]] for a in legal_actions}
        return p, v

    def forward(self, s: torch.Tensor):
        raise NotImplementedError

    def predict_batch(self, ss: list):  # TODO -- move up (to Model interface)
        pass  # TODO -- implement

    def forward_batch(self, ss: torch.Tensor):
        pass  # TODO

    def fit(self, examples: list):
        pass  # TODO

    def deepcopy(self):
        raise NotImplementedError

    def save(self, directory_path: str, examples: list = None):
        pass  # TODO

    @staticmethod
    def load(directory_path: str):
        pass


class ResidualNetwork(TorchModel):  # TODO

    def __init__(self, game: callable, args: argparse.Namespace):
        assert args.num_res_blocks
        assert args.c_l2reg  # TODO
        assert args.conv_block_filters
        assert args.conv_block_kernel_size
        assert args.conv_block_strides
        super().__init__(game, args)

    def state_input(self, s: GameState):
        raise NotImplementedError

    def forward(self, s: torch.Tensor):
        pass

    def deepcopy(self):
        pass


class ConvBlock(torch.nn.Module):  # TODO

    def forward(self, x):
        pass


class ResBlock(torch.nn.Module):

    def forward(self, x):  # TODO
        pass


class PolicyHead(torch.nn.Module):  # TODO

    def forward(self, x):
        pass


class ValueHead(torch.nn.Module):  # TODO

    def forward(self, x):
        pass

