import numpy as np
import pickle
import os
from tqdm import trange

import torch
import torch.nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import argparse
from functools import reduce

from alphazero.game import GameState
from alphazero.model import Model


class TorchModel(Model, torch.nn.Module):

    """
    Abstract class defining a model based on the PyTorch framework
    """

    def __init__(self, game: callable, args: argparse.Namespace):
        """
        Create a new TorchModel
        :param game: The constructor of the game that should be played
        :param args: Parsed arguments containing hyperparameters
                        - disable_cuda: flag that indicates that the GPU should not be used
                        - batch_size: batch size that is used when training the model
                        - epochs: number of epochs that are performed when training the model
        """
        torch.nn.Module.__init__(self)
        super().__init__(game, args)

        self.in_shape = game.board_shape()  # Shape of the input image. NOTE: this does not include number of channels!
        self.action_space = game.action_space()  # All possible actions that could be played in the game
        self.out_size = len(self.action_space)  # Size of the action space

        self.action_index = {a: i for i, a in enumerate(self.action_space)}  # Map all possible actions to some index

        self.use_cuda = not args.disable_cuda and torch.cuda.is_available()

    def state_input(self, s: GameState):
        raise NotImplementedError

    def get_optimizer(self):
        raise NotImplementedError

    def forward(self, s: torch.Tensor):
        raise NotImplementedError

    def predict(self, s: GameState) -> tuple:
        """
        Perform a forward pass through the network to obtain a (policy, value)-pair for the given game state
        :param s: The game state for which a policy and value should be obtained
        :return: a two-tuple consisting of
                    - A probability distribution over all legal actions (policy) from the given game state.
                      The distribution is contained in a dict with actions as keys and floats as their corresponding
                      probabilities.
                    - A float value representing the estimated probability of winning the game
        """
        # This function is implemented as a special case of predict_batch with batch size 1
        [p], [v] = self.predict_batch([s])
        return p, v

    def predict_batch(self, ss: list) -> tuple:
        """
        Perform a forward pass to obtain (policy, value)-pairs for the given list of game states
        :param ss: A list of GameState objects for which the policies and values should be estimated
        :return: a two-tuple consisting of
                    - A list of policies. The policies correspond to the game states by index in the list
                    - A list of values. The values correspond to the game states by index in the list
                For a description of what the policies and values consist of see self.predict
        """
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
        ps *= mask
        # Normalize the probability distribution so it sums to 1 again
        ps /= torch.sum(ps, dim=1)
        # Return the probability distribution as dict
        ps = [{a: ps[i][self.action_index[a]].item() for a in actions}
              for i, actions in enumerate(legal_actions)]
        vs = [v.item() for v in vs]
        return ps, vs

    def fit(self, examples: list):
        """
        Train the model on a list of examples

        The model is trained by performing a forward pass for each game state in the examples which results in some
        (policy, value) pair. Then the loss is computed by taking the MSE loss between the predicted value and the true
        outcome of the game as given by the example. The loss for the policy is computed by taking the cross entropy of
        the predicted policy and the policy obtained from the example (which was obtained from performing a number of
        Monte Carlo Tree Searches). These loss values are used to obtain a gradient which is used to update the model
        parameters.
        :param examples: A list of examples on which the network should be trained. One example consists of:
                            - The current game state (as GameState object)
                            - The distribution from which a move was sampled
                            - The win/loss value resulting from that move
        """
        # Check if cpu of gpu should be used
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        # Set Module to train mode
        self.train()
        # Create a shuffled list of examples
        examples = list(examples)
        np.random.shuffle(examples)
        # Prepare examples for training
        # Note: Examples is a list of 3-tuples. Each 3-tuple contains
        #   - The current game state (as GameState object)
        #   - The distribution from which a move was sampled
        #   - The win/loss value resulting from that move
        # All data should be converted to tensors to be suitable network input
        input_states, target_dists, target_values = [], [], []
        for s, d, v in examples:
            # Convert all GameState objects to Tensors suitable for input
            input_states += [self.state_input(s).unsqueeze(0)]  # unsqueeze tensor to create batch dimension
            # Convert all target values to Tensors
            target_values += [torch.Tensor([v]).unsqueeze(0).to(device)]
            # Convert all target distributions to Tensors
            dist = torch.zeros(self.out_size, device=device)
            # The index of each value in the distribution determines which action the value corresponds to
            for a, p in d.items():
                dist[self.action_index[a]] = p  # Set each probability at the proper index
            target_dists += [dist.unsqueeze(0)]

        # Concatenate all example entries to three large data tensors
        input_states = torch.cat(input_states, dim=0)  # One for input states
        target_dists = torch.cat(target_dists, dim=0)  # One for target distributions
        target_values = torch.cat(target_values, dim=0)  # One for target values

        # Create a dataset containing all examples
        dataset = TensorDataset(input_states, target_dists, target_values)
        # Create a data loader for sampling from the data set
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        # Define the loss functions
        # The policy head output has a cross entropy loss function
        # The default PyTorch CE loss cant be used as it only works with one-hot vector labels of type Long
        cce = lambda y_p, y_t: torch.mean(-torch.sum(torch.mul(y_t, torch.log(y_p)), dim=1))
        # The value head output has a mean squared error loss function
        mse = torch.nn.MSELoss()

        # Create a progress bar for iterating through the data set for some number of epochs
        train_iter = trange(1, self.args.epochs + 1,
                            desc='Epoch 1 | Batch 0/{} | P loss <?> | V loss <?>'.format(len(data_loader)),
                            leave=True)
        # Iterate through the data set for some number of epochs
        for epoch in train_iter:
            # Iterate through all batches of examples in the data set
            for i, (s_batch, d_batch, v_batch) in enumerate(data_loader):
                # Perform a forward pass through the network
                ps, vs = self.forward(s_batch)
                # Compute the losses on the network's outputs
                policy_loss = cce(ps, d_batch)
                value_loss = mse(vs, v_batch)
                # Reset the gradients of previous iterations to zero
                self.get_optimizer().zero_grad()
                # Perform a backward pass to compute the gradients for this iteration
                (policy_loss + value_loss).backward()
                # Do an optimization step using the computed gradients
                self.get_optimizer().step()

                # Update the progress bar
                train_iter.set_description(
                    'Epoch {:3d} | Batch {}/{} | P loss {:.3f} | V loss {:.3f}'.format(
                        epoch,
                        i + 1,
                        len(data_loader),
                        policy_loss.item(),
                        value_loss.item(),
                    )
                )

    def deepcopy(self):
        """
        Create a deep copy of this model
        :return: a deep copy of this model
        """
        model_copy = type(self)(self.game, self.args)
        model_copy.load_state_dict(self.state_dict())
        model_copy.get_optimizer().load_state_dict(self.get_optimizer().state_dict())
        return model_copy

    def save(self, directory_path: str, examples: list = None):
        """
        Save this model in the specified directory
        :param directory_path: The path to the directory in which the model should be saved
        :param examples: Optional list of training examples that can be saved with the model
        """
        if not os.path.isdir(directory_path):  # Ensure the directory exists
            os.mkdir(directory_path)
        # Collect relevant data that should be saved
        checkpoint = {
            'model': self,
            'state': self.state_dict(),
            'optimizer_state': self.get_optimizer().state_dict(),
        }
        # Save the model
        with open(directory_path + '/model.pth', 'wb') as f:
            torch.save(checkpoint, f)
        # Optionally save the examples
        if examples:
            with open(directory_path + '/examples.pickle', 'wb') as f:
                pickle.dump(examples, f)

    @staticmethod
    def load(directory_path: str):
        """
        Load a model from the specified directory
        :param directory_path: The path to the directory from which the model should be loaded
        :return: a two-tuple consisting of
                    - The loaded model
                    - Optional examples that were stored with the model. (empty list if non-existent)
        """
        checkpoint = torch.load(directory_path + '/model.pth')
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state'])
        model.get_optimizer().load_state_dict(checkpoint['optimizer_state'])
        if os.path.isfile(directory_path + '/examples.pickle'):
            with open(directory_path + '/examples.pickle', 'rb') as f:
                examples = pickle.load(f)
        else:
            examples = []
        return model, examples


class ResidualNetwork(TorchModel):

    """
    Residual Neural Network architecture for playing board games
    The network architecture consists of
        - One convolutional block
        - Some number of residual blocks
        - A policy head giving a probability distribution over all actions in the action space
        - A value head giving an estimate of the probability of winning the game

        All these components are defined as separate PyTorch Modules

    """

    def __init__(self, game: callable, args: argparse.Namespace):
        """
        Create a new Residual Neural Network
        :param game: The constructor of the game the network should play
        :param args: Parsed arguments containing hyperparameters
                        - input_channels: Number of channels of the network input
                        - num_res_blocks: Number of residual blocks in the network
                        - c_l2reg: Parameter controlling the influence of weight decay
                        - block_channels: Number of channels in the conv/resd blocks
                        - conv_block_kernel_size: Kernel size in the convolutional block
                        - conv_block_padding: Amount of padding in the convolutional block
                        - res_block_kernel_size: Kernel size in the residual blocks
                        - res_block_padding: Amount of padding in the residual block

        Let (bs, ci, cb, w, h, out) denote
          - the batch size
          - the number of input channels
          - the number of block channels
          - the width of the input image
          - the height of the input image
          - the size of the game's action space,
        respectively. These will be used to denote the shape of the Tensors in the comments

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
                                         weight_decay=args.c_l2reg)

    def state_input(self, s: GameState):
        raise NotImplementedError

    def get_optimizer(self):
        """
        :return: the optimizer used for training this model
        """
        return self.optimizer

    def forward(self, x: torch.Tensor):
        """
        A forward pass through the network
        :param x: Input tensor of shape: bs, ci, w, h
        :return: a two-tuple consisting of
                    - a policy tensor containing a probability distribution over the entire action space
                    - a value tensor containing an estimate of the probability of winning from the input gamestate
        """
        x = self.conv_block(x)              # Shape: bs, cb, w, h
        for res_block in self.res_blocks:
            x = res_block(x)                # Shape: bs, cb, w, h
        p = self.p_head(x)                  # Shape: bs, out
        v = self.v_head(x)                  # Shape: bs, 1
        return p, v


class ConvBlock(torch.nn.Module):

    def __init__(self, args: argparse.Namespace):
        """
        A convolutional block consists of
            - A convolutional layer
            - A batch normalization layer
            - Relu activation

        Let (bs, ci, cb, w, h) denote
          - the batch size
          - the number of input channels
          - the number of block channels
          - the width of the input image
          - the height of the input image

        :param args: Parsed arguments containing hyperparameters
                        - input_channels: The number of channels in the network's input
                        - block_channels: The number of channels in the conv/resd blocks
                        - conv_block_kernel_size: The kernel size of the convolutional layers
                        - conv_block_padding: The amount of padding in the convolutional layers
        """
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=args.input_channels,
                                      out_channels=args.block_channels,
                                      kernel_size=args.conv_block_kernel_size,
                                      stride=1,
                                      padding=args.conv_block_padding,
                                      )
        self.batchnorm = torch.nn.BatchNorm2d(num_features=args.block_channels)

    def forward(self, x):
        """
        A forward pass through the convolutional block
        :param x: An input Tensor. Shape: bs, ci, w, h
        :return: An output Tensor. Shape: bs, cb, w, h
        """
        x = self.conv2d(x)      # Shape: bs, cb, w, h
        x = self.batchnorm(x)   # Shape: bs, cb, w, h
        return F.relu(x)        # Shape: bs, cb, w, h


class ResBlock(torch.nn.Module):

    def __init__(self, args: argparse.Namespace):
        """
        A Residual Block consists of
            - A convolutional layer
            - A batch normalization layer
            - Relu activation
            - A convolutional layer
            - A batch normalization layer
            - An operation where the original input of the block is added to the output of the previous layer
            - Relu activation

        Let (bs, ci, cb, w, h) denote
          - the batch size
          - the number of input channels
          - the number of block channels
          - the width of the input image
          - the height of the input image
        respectively. These will be used to denote the shape of the Tensors in the comments

        :param args: Parsed arguments containing hyperparameters
                        - block_channels: The number of channels in the conv/resd blocks
                        - res_block_kernel_size: The kernel size of the convolutional layers
                        - res_block_padding: The amount of padding in the convolutional layers
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
        A forward pass through the residual block
        :param x: An input Tensor. Shape: bs, cb, w, h
        :return: An output Tensor. Shape: bs, cb, w, h
        """
        x_in = x                    # Keep a reference to the original block input. Shape: bs, cb, w, h
        x = self.conv2d_1(x)        # Shape: bs, cb, w, h
        x = self.batchnorm_1(x)     # Shape: bs, cb, w, h
        x = F.relu(x)               # Shape: bs, cb, w, h
        x = self.conv2d_2(x)        # Shape: bs, cb, w, h
        x = self.batchnorm_2(x)     # Shape: bs, cb, w, h
        x += x_in                   # Add the original input to the output of the previous layer. Shape: bs, cb, w, h
        return F.relu(x)            # Shape: bs, cb, w, h


class PolicyHead(torch.nn.Module):

    def __init__(self, in_shape: tuple, out_size: int, args: argparse.Namespace):
        """
        The policy head consists of
            - A convolutional layer with two output channels and a kernel size of (1, 1)
            - A batch normalization layer
            - Relu activation
            - A dense layer taking in a flattened version of the previous layer's output tensor.
              Outputs a value for each possible action in the game's action space
            - Softmax activation that converts the dense layer's output into a probability distribution over the game's
              action space

        Let (bs, ci, cb, w, h, out) denote
          - the batch size
          - the number of input channels
          - the number of block channels
          - the width of the input image
          - the height of the input image
          - the size of the game's action space,
        respectively. These will be used to denote the shape of the Tensors in the comments

        :param in_shape: The shape of the input data
        :param out_size: The size of the action space
        :param args: Parsed arguments containing hyperparameters
                        - block_channels: The number of channels in the conv/resd blocks

        """
        super().__init__()
        channels = 2  # Number of output channels of the conv layer
        self.conv2d = torch.nn.Conv2d(in_channels=args.block_channels,
                                      out_channels=channels,
                                      kernel_size=(1, 1),
                                      stride=1,
                                      )
        self.batchnorm = torch.nn.BatchNorm2d(num_features=channels)
        self.dense = torch.nn.Linear(in_features=channels * reduce(lambda x, y: x * y, in_shape[1:]),
                                     out_features=out_size,  # Output is over the entire action space
                                     bias=True)

    def forward(self, x):
        """
        A forward pass through the value head
        :param x: An input Tensor. Shape: bs, cb, w, h
        :return: An output Tensor, containing the output policy of the network (over the entire action space)
        """
        x = self.conv2d(x)          # Conv layer has 2 output channels -> Shape: bs, 2, w, h
        x = self.batchnorm(x)       # Shape: bs, 2, w, h
        x = F.relu(x)               # Shape: bs, 2, w, h
        x = x.view(x.shape[0], -1)  # Flatten the image -> Shape: bs, w * h
        x = self.dense(x)           # Shape: bs, out
        return F.softmax(x, dim=1)  # Shape: bs, out


class ValueHead(torch.nn.Module):

    def __init__(self, in_shape: tuple, args: argparse.Namespace):
        """
        The value head consists of
            - A convolutional layer with one output channel and a kernel size of (1, 1)
            - A batch normalization layer
            - Relu activation
            - A Dense layer taking a flattened version of the output of the Relu activation. Has 32 output nodes
            - Relu activation
            - A Dense layer with 32 input nodes and one output node
            - Tanh activation to give the final value value

        Let (bs, ci, cb, w, h) denote
          - the batch size
          - the number of input channels
          - the number of block channels
          - the width of the input image
          - the height of the input image,
        respectively. These will be used to denote the shape of the Tensors in the comments

        :param in_shape: The shape of the input data
        :param args: Parsed arguments containing hyperparameters
                        - block_channels: The number of channels in the conv/resd blocks
        """
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=args.block_channels,
                                      out_channels=1,
                                      kernel_size=(1, 1),
                                      stride=1,
                                      )
        self.batchnorm = torch.nn.BatchNorm2d(num_features=1)
        self.dense_1 = torch.nn.Linear(in_features=reduce(lambda x, y: x * y, in_shape[1:]),  # input shape (bs, w * h)
                                       out_features=32,
                                       bias=True)
        self.dense_2 = torch.nn.Linear(in_features=32,
                                       out_features=1,
                                       bias=True)

    def forward(self, x):
        """
        A forward pass through the value head
        :param x: An input Tensor. Shape: bs, cb, w, h
        :return: An output Tensor, containing the output value of the network
        """
        x = self.conv2d(x)          # Conv layer has 1 output channel -> Shape: bs, 1, w, h
        x = self.batchnorm(x)       # Shape: bs, 1, w, h
        x = F.relu(x)               # Shape: bs, 1, w, h
        x = x.view(x.shape[0], -1)  # Flatten the image -> Shape: bs, w * h
        x = self.dense_1(x)         # Shape: bs, 32
        x = F.relu(x)               # Shape: bs, 32
        x = self.dense_2(x)         # Shape: bs, 1
        return torch.tanh(x)        # Shape: bs, 1


if __name__ == '__main__':
    from alphazero.games.tictactoe.tictactoe import TicTacToe

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

    # print(torch.nn.ModuleList(_resnet.modules()))
    # print(torch.nn.ParameterList(_resnet.parameters()))
