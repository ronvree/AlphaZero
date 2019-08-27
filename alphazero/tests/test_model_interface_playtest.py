import torch
import torch.nn
import torch.nn.functional as func

import argparse

from functools import reduce

from alphazero.games.tictactoe.tictactoe import TicTacToe
from alphazero.model_pytorch import TorchModel


class TestModel(TorchModel):

    def __init__(self, args: argparse.Namespace):
        super().__init__(TicTacToe, args)
        self.linear = torch.nn.Linear(in_features=reduce(lambda x, y: x * y, self.in_shape),
                                      out_features=self.out_size)
        self.policy_head = torch.nn.Linear(in_features=self.out_size,
                                           out_features=self.out_size)
        self.value_head = torch.nn.Linear(in_features=self.out_size,
                                          out_features=1)

    def state_input(self, s: TicTacToe):
        return torch.Tensor(s.grid)

    def forward(self, s: torch.Tensor):
        s = s.flatten()
        s = func.relu(self.linear(s))
        p = func.softmax(self.policy_head(s), dim=0)
        v = torch.tanh(self.value_head(s))
        return p, v

    def deepcopy(self):
        pass


if __name__ == '__main__':
    import numpy.random

    _args = argparse.Namespace()

    _state = TicTacToe(_args)

    _model = TestModel(_args)

    print(_state)
    while not _state.is_terminal():
        _p, _v = _model.predict(_state)

        r = numpy.random.random()
        total = 0
        for a, p in _p.items():
            total += p
            if r < total:
                _a = a
                break

        _state.do_move(_a)

        print(_p)
        print(_v)
        print(_a)
        print(_state)
