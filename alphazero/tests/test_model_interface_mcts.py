import argparse
import torch
import torch.nn
import torch.nn.functional as func
from functools import reduce

from alphazero.model_pytorch import TorchModel
from alphazero.montecarlo import MCST
from alphazero.games.tictactoe.tictactoe import TicTacToe


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

    _args = argparse.Namespace()
    _args.c_puct = 1
    _args.epsilon = 0.25
    _args.alpha = 0.03

    _tree = MCST(_args)

    _s = TicTacToe(_args)
    _m = TestModel(_args)

    while not _s.is_terminal():
        # _tree.clear()
        for _ in range(100):
            _tree.search(_s, _m)
        _a, _pi = _tree.action(_s, temperature=0.5)
        _s.do_move(_a)

    print(_s)
