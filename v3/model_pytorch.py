import numpy as np
import pickle
import os

import torch
import torch.nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import argparse
from functools import reduce

from v3.game import GameState
from v3.model import Model


class TorchModel(Model, torch.nn.Module):

    def __init__(self, game: callable, args: argparse.Namespace):
        torch.nn.Module.__init__(self)
        super().__init__(game, args)
        self.args = args

        self.in_shape = game.board_shape()
        self.action_space = game.action_space()
        self.out_size = len(self.action_space)

        self.action_index = {a: i for i, a in enumerate(self.action_space)}

        self.use_cuda = not args.disable_cuda and torch.cuda.is_available()

    def state_input(self, s: GameState):
        raise NotImplementedError

    def get_optimizer(self):
        raise NotImplementedError

    def forward(self, s: torch.Tensor):
        raise NotImplementedError

    def predict(self, s: GameState) -> tuple:
        [p], [v] = self.predict_batch([s])
        return p, v

    def predict_batch(self, ss: list):  # TODO -- move up (to Model interface)
        # Check if cpu of gpu should be used
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        # Set Module to eval mode
        self.eval()
        # Convert GameState object to suitable input
        states = torch.cat([self.state_input(s).unsqueeze(0) for s in ss], dim=0)
        # Obtain predictions of the network
        ps, vs = self.forward(states)  # p is a policy over all moves of the game
        # Mask all illegal actions
        legal_actions = [s.get_actions() for s in ss]
        mask = torch.zeros((len(ss), self.out_size), device=device)
        for i, actions in enumerate(legal_actions):
            for a in actions:
                mask[i][self.action_index[a]] = 1
        ps *= mask  # TODO -- check if hadamard product
        # Normalize the probability distribution so it sums to 1 again
        ps /= torch.sum(ps, dim=1)  # TODO -- check if element-wise
        # Return the probability distribution as dict
        ps = [{a: ps[i][self.action_index[a]].item() for a in actions}
              for i, actions in enumerate(legal_actions)]
        vs = [v.item() for v in vs]
        return ps, vs

    def fit(self, examples: list):

        # Note to self: Examples is a list of 3-tuples. Each 3-tuple contains
        #   - The current game state (as GameState object)
        #   - The distribution from which a move was sampled
        #   - The win/loss value resulting from that move

        # Check if cpu of gpu should be used
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        # Set Module to train mode
        self.train()
        # Create a shuffled list of examples
        examples = list(examples)
        np.random.shuffle(examples)
        # Prepare examples for training
        input_states, target_dists, target_values = [], [], []
        for s, d, v in examples:
            input_states += [self.state_input(s).unsqueeze(0)]
            target_values += [torch.Tensor([v]).unsqueeze(0).to(device)]

            dist = torch.zeros(self.out_size, device=device)
            for a, p in d.items():
                dist[self.action_index[a]] = p  # TODO -- does this keep the gradient?
            target_dists += [dist.unsqueeze(0)]              # TODO -- not sure, do properly

        input_states = torch.cat(input_states, dim=0)
        target_dists = torch.cat(target_dists, dim=0)
        target_values = torch.cat(target_values, dim=0)

        dataset = TensorDataset(input_states, target_dists, target_values)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        cce = lambda y_p, y_t: torch.mean(-torch.sum(torch.mul(y_t, torch.log(y_p)), dim=1))
        mse = torch.nn.MSELoss()

        for epoch in range(1, self.args.epochs + 1):

            for i, (s_batch, d_batch, v_batch) in enumerate(data_loader):

                ps, vs = self.forward(s_batch)
                policy_loss = cce(ps, d_batch)
                value_loss = mse(vs, v_batch)

                self.get_optimizer().zero_grad()
                (policy_loss + value_loss).backward()
                self.get_optimizer().step()

                print('Epoch {:3d} | Batch {}/{} | P loss {:.3f} | V loss {:.3f}'.format(
                    epoch,
                    i,
                    len(data_loader.dataset) // self.args.batch_size,
                    policy_loss.item(),
                    value_loss.item(),
                ))
                # TODO -- performance metric
                pass  # TODO -- logging

    def deepcopy(self):
        raise NotImplementedError

    def save(self, directory_path: str, examples: list = None):
        """

        :param directory_path:
        :param examples:
        :return:
        """
        checkpoint = {
            'model': self,
            'state': self.state_dict(),
            'optimizer_state': self.get_optimizer().state_dict(),
        }
        with open(directory_path + '/model.pth', 'wb') as f:
            torch.save(checkpoint, f)

        if examples:
            with open(directory_path + '/examples.pickle', 'wb') as f:
                pickle.dump(examples, f)

    @staticmethod
    def load(directory_path: str, load_examples: bool = False):
        """

        :param directory_path:
        :param load_examples:
        :return:
        """
        checkpoint = torch.load(directory_path + '/model.pth')
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state'])
        model.get_optimizer().load_state_dict(checkpoint['optimizer_state'])
        if load_examples and os.path.isfile(directory_path + '/examples.pickle'):
            with open(directory_path + '/examples.pickle', 'rb') as f:
                examples = pickle.load(f)
        else:
            examples = None
        return model, examples


class ResidualNetwork(TorchModel):

    """

    """

    def __init__(self, game: callable, args: argparse.Namespace):
        """

        :param game:
        :param args:
        """
        assert args.input_channels
        assert args.num_res_blocks
        assert args.c_l2reg
        assert args.block_channels
        assert args.conv_block_kernel_size
        assert args.conv_block_padding
        assert args.res_block_kernel_size
        assert args.res_block_padding
        super(ResidualNetwork, self).__init__(game, args)

        self.conv_block = ConvBlock(args)

        self.res_blocks = []
        for i in range(args.num_res_blocks):
            block = ResBlock(args)
            self.add_module('res_block_{}'.format(i), block)

        self.p_head = PolicyHead((args.block_channels,) + self.in_shape, self.out_size, args)
        self.v_head = ValueHead((args.block_channels,) + self.in_shape, args)

        if self.use_cuda:
            self.cuda()

        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=self.args.learning_rate,
                                         momentum=self.args.momentum,
                                         weight_decay=args.c_l2reg)  # TODO -- check if correct

    def state_input(self, s: GameState):
        raise NotImplementedError

    def get_optimizer(self):
        """

        :return:
        """
        return self.optimizer

    def forward(self, x: torch.Tensor):
        """

        :param x:
        :return:
        """
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        p = self.p_head(x)
        v = self.v_head(x)
        return p, v

    def deepcopy(self):
        pass  # TODO


class ConvBlock(torch.nn.Module):

    def __init__(self, args: argparse.Namespace):
        """

        :param args:
        """
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=args.input_channels,  # TODO -- mention in docs
                                      out_channels=args.block_channels,
                                      kernel_size=args.conv_block_kernel_size,
                                      stride=1,
                                      padding=args.conv_block_padding,
                                      )
        self.batchnorm = torch.nn.BatchNorm2d(num_features=args.block_channels)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv2d(x)
        x = self.batchnorm(x)
        return F.relu(x)


class ResBlock(torch.nn.Module):

    def __init__(self, args: argparse.Namespace):
        """

        :param args:
        """
        super().__init__()

        self.conv2d_1 = torch.nn.Conv2d(in_channels=args.block_channels,
                                        out_channels=args.block_channels,
                                        kernel_size=args.res_block_kernel_size,
                                        stride=1,
                                        padding=args.res_block_padding,
                                        )
        self.batchnorm_1 = torch.nn.BatchNorm2d(num_features=args.block_channels)

        self.conv2d_2 = torch.nn.Conv2d(in_channels=args.block_channels,
                                        out_channels=args.block_channels,
                                        kernel_size=args.res_block_kernel_size,
                                        stride=1,
                                        padding=args.res_block_padding,
                                        )
        self.batchnorm_2 = torch.nn.BatchNorm2d(num_features=args.block_channels)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x_in = x
        x = self.conv2d_1(x)
        x = self.batchnorm_1(x)
        x = F.relu(x)
        x = self.conv2d_2(x)
        x = self.batchnorm_2(x)
        x += x_in
        return F.relu(x)


class PolicyHead(torch.nn.Module):

    def __init__(self, in_shape: tuple, out_size: int, args: argparse.Namespace):
        """

        :param in_shape:
        :param out_size:
        :param args:
        """
        super().__init__()
        channels = 2
        self.conv2d = torch.nn.Conv2d(in_channels=args.block_channels,
                                      out_channels=channels,
                                      kernel_size=(1, 1),
                                      stride=1,
                                      )
        self.batchnorm = torch.nn.BatchNorm2d(num_features=channels)
        self.dense = torch.nn.Linear(in_features=channels * reduce(lambda x, y: x * y, in_shape[1:]),
                                     out_features=out_size,
                                     bias=True)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.dense(x.view(x.shape[0], -1))
        return F.softmax(x, dim=1)


class ValueHead(torch.nn.Module):

    def __init__(self, in_shape: tuple, args: argparse.Namespace):
        """

        :param in_shape:
        :param args:
        """
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=args.block_channels,
                                      out_channels=1,
                                      kernel_size=(1, 1),
                                      stride=1,
                                      )
        self.batchnorm = torch.nn.BatchNorm2d(num_features=1)
        self.dense_1 = torch.nn.Linear(in_features=reduce(lambda x, y: x * y, in_shape[1:]),
                                       out_features=32,
                                       bias=True)
        self.dense_2 = torch.nn.Linear(in_features=32,
                                       out_features=1,
                                       bias=True)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.dense_1(x.view(x.shape[0], -1))
        x = F.relu(x)
        x = self.dense_2(x)
        return torch.tanh(x)


if __name__ == '__main__':
    from v3.games.tictactoe.tictactoe import TicTacToe

    _args = argparse.Namespace()
    _args.input_channels = 3

    _bs, _c, _w, _h = 10, 3, 5, 4
    _in_shape = (_c, _w, _h)

    _x = torch.randn(_bs, _c, _w, _h)

    _args.block_channels = 16
    _args.conv_block_kernel_size = (3, 3)
    _args.conv_block_padding = 1

    _conv_block = ConvBlock(_args)

    print('Applying Conv Block')
    _x = _conv_block(_x)
    print(_x.shape)
    # print(_x)

    _args.res_block_kernel_size = (3, 3)
    _args.res_block_padding = 1

    _res_block_1 = ResBlock(_args)
    _res_block_2 = ResBlock(_args)
    _res_block_3 = ResBlock(_args)

    print('Applying Res Block 1')
    _x = _res_block_1(_x)
    print(_x.shape)
    # print(_x)

    print('Applying Res Block 2')
    _x = _res_block_2(_x)
    print(_x.shape)
    # print(_x)

    print('Applying Res Block 3')
    _x = _res_block_3(_x)
    print(_x.shape)
    # print(_x)

    _out_size = 6

    _p_head = PolicyHead(_in_shape, _out_size, _args)
    _v_head = ValueHead(_in_shape, _args)

    print('Applying Policy Head')
    _p = _p_head(_x)
    print(_p.shape)
    # print(_p)

    print('Applying Value Head')
    _v = _v_head(_x)
    print(_v.shape)
    # print(_v)

    _args.input_channels = 1
    _args.block_channels = 16
    _args.conv_block_kernel_size = (3, 3)
    _args.conv_block_padding = 1
    _args.res_block_kernel_size = (3, 3)
    _args.res_block_padding = 1

    _args.num_res_blocks = 3
    _args.c_l2reg = 0.001
    _args.conv_block_kernel_size = (3, 3)
    _args.conv_block_padding = 1
    _args.res_block_kernel_size = (3, 3)
    _args.res_block_padding = 1
    _args.disable_cuda = True
    _args.learning_rate = 0.01
    _args.momentum = 0.9

    _resnet = ResidualNetwork(TicTacToe, _args)
    _resnet.state_input = lambda s: torch.Tensor(s.grid).view(1, *s.grid.shape)

    _game = TicTacToe(_args)

    print('Applying ResNet fw pass')
    _p, _v = _resnet.predict(_game)
    print(_p)
    print(_v)
