import numpy as np
import keras as ks
from keras import Input
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, add, Flatten
from keras.optimizers import SGD
from keras.models import model_from_json

import argparse
import os
import pickle

from alphazero.game import GameState
from alphazero.model import Model


class NeuralNetwork(Model):
    """
        Model making use of a Keras model to make predictions
    """

    def __init__(self, game: callable, args: argparse.Namespace):
        """
        Wrapper to make a Keras model compatible with this AlphaZero implementation
        :param game: The type of game that will be played by this model. (Just pass the constructor)
        :param args: Parsed arguments containing hyperparameters of the model
        """
        super().__init__(game, args)

        self.action_space = game.action_space()

        self.input_shape = game.board_shape() + (1,)
        self.output_size = len(self.action_space)

        self.action_index = {a: self.action_space.index(a) for a in self.action_space}

    def get_model(self) -> ks.Model:
        """
        :return: Keras model that makes (p, v) predictions
        """
        raise NotImplementedError

    def state_input(self, s: GameState):
        """
        Converts a game state to suitable input for the network
        :param s: The input state
        :return: an input state representation that is compatible with the network
                 Note: Output shape should be self.input_shape!
        """
        raise NotImplementedError

    def predict(self, s: GameState):
        """
        Predict a (p, v)-pair by giving the state to the neural network, where
            p is a dictionary containing a probability distribution over all legal actions
            v is a scalar value estimating the probability of winning from state s
        :param s: The input state for the network
        :return: The predicted (p, v)-tuple
        """
        state = self.state_input(s)
        state = np.reshape(state, (1,) + state.shape)  # Reshape input to a list of a single data point

        [p_all], [[v]] = self.get_model().predict(state)  # Obtain predictions of the network

        # Mask all illegal actions
        legal_actions = s.get_actions()
        mask = np.zeros(self.output_size)
        for a in legal_actions:
            mask[self.action_index[a]] = 1
        p_all *= mask

        # Normalize the distribution so the probabilities sum to 1
        p_all /= p_all.sum()

        # Get the distribution over all legal actions
        p = {a: p_all[self.action_index[a]] for a in legal_actions}

        return p, v

    def predict_batch(self, ss: list) -> tuple:
        pass  # TODO

    def fit(self, examples: list):
        """
        Train the Keras model on the given examples
        :param examples: A list of examples that is used to train the model
        """
        # Prepare all examples for training
        input_boards, target_pis, target_vs = [], [], []
        for s, pi, v in examples:
            # Reshape the state observation to a shape suitable for convolution
            # input_boards.append(np.reshape(s, self.input_shape))
            input_boards.append(self.state_input(s))

            # Map all actions to a fixed index
            pi_ = np.zeros(self.output_size)
            for a, p in pi.items():
                pi_[self.action_index[a]] = p
            target_pis.append(pi_)

            # Add reward value
            target_vs.append(v)

        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        # Train the model on the examples
        self.get_model().fit(x=input_boards,
                             y=[target_pis, target_vs],
                             batch_size=self.args.batch_size,
                             epochs=self.args.epochs,
                             verbose=self.args.verbose)

    def save(self, directory_path: str, examples: list = None):
        """
        Save the model to some specified directory
        The model is saved as:
            - model.json -- containing the model architecture
            - weights.h5 -- containing the model weights
            - examples.pickle -- containing training examples saved with the model (optional)
        """
        try:
            os.mkdir(directory_path)
        except FileExistsError:
            pass
        # Save the model weights
        self.get_model().save_weights(directory_path + '/weights.h5')
        # Save the model architecture
        with open(directory_path + '/model.json', 'w') as f:
            f.write(self.get_model().to_json())
        # If examples were given, pickle them
        if examples:
            pickle.dump(examples, open(directory_path + '/examples.pickle', 'wb'))

    @staticmethod
    def load(directory_path: str) -> tuple:  # TODO -- find a way to load the ks model into a model wrapper
        """
        Load a model from the specified directory
        :param directory_path: The directory from which the model should be loaded
        :return: A two-tuple of:
                    - The loaded model
                    - Optional list of training examples that were saved with the model
        """
        assert os.path.isfile(directory_path + '/model.json')  # Make sure the required files actually exist
        assert os.path.isfile(directory_path + '/weights.h5')

        with open(directory_path + '/model.json', 'r') as model_f:
            # Re-create the model architecture from the json file
            model = model_from_json(model_f.read())
            # Set the model weights
            model.load(directory_path + '/weights.h5')
            # Check if any training examples were saved with the model. If so, unpickle them
            if os.path.isfile(directory_path + '/examples.pickle'):
                examples = pickle.load(open(directory_path + '/examples.pickle', 'rb'))
            else:
                examples = None
            return model, examples

    def deepcopy(self):
        raise NotImplementedError


class ResidualNeuralNetwork(NeuralNetwork):
    """
        A Residual Convolutional Neural Network as described by the AlphaZero paper
    """

    def __init__(self,
                 game: callable,
                 args: argparse.Namespace,
                 num_resd_block: int,
                 compile_parameters: dict = None,
                 conv_block_params=None,
                 resd_block_params=None,):
        """
        Create a new Residual Neural Network
        :param game: The type of game that will be played by this model. (Just pass the constructor)
        :param num_resd_block: The number of residual blocks in the network
        :param compile_parameters: Dictionary that allows arguments to be passed to the model compile function
        :param conv_block_params: Dictionary that allows arguments to be passed to the convolutional block
        :param resd_block_params: Dictionary that allows arguments to be passed to the residual blocks
        """
        super().__init__(game, args)

        #######################################
        # Set default parameters of the model #
        #######################################

        self.c_L2reg = args.c_l2reg

        # Create Convolutional Block parameters
        if conv_block_params is None:
            self.conv_block_params = {}
        else:
            self.conv_block_params = conv_block_params
        # Set default parameters if not already existent
        self.conv_block_params.setdefault('filters', 64)
        self.conv_block_params.setdefault('kernel_size', (3, 3))
        self.conv_block_params.setdefault('strides', 1)
        self.conv_block_params.setdefault('kernel_regularizer', ks.regularizers.l2(self.c_L2reg))

        # Create Residual Block parameters
        if resd_block_params is None:
            self.resd_block_params = {}
        else:
            self.resd_block_params = resd_block_params
        # Set default parameters if not already existent
        self.resd_block_params.setdefault('filters', 64)
        self.resd_block_params.setdefault('kernel_size', (3, 3))
        self.resd_block_params.setdefault('strides', 1)
        self.resd_block_params.setdefault('kernel_regularizer', ks.regularizers.l2(self.c_L2reg))

        # Create model compilation parameters
        if compile_parameters is None:
            self.compile_parameters = {}
        else:
            self.compile_parameters = compile_parameters
        # Set default parameters if not already existent
        self.compile_parameters.setdefault('loss', ['categorical_crossentropy', 'mean_squared_error'])
        self.compile_parameters.setdefault('optimizer', SGD(lr=self.args.learning_rate, momentum=self.args.momentum))

        #####################
        # Build the network #
        #####################

        # Create input layer
        self.input_layer = Input(shape=self.input_shape)
        x = self.input_layer
        # A single convolutional block follows the input layer
        x = self.convolutional_block(x)

        # A sequence of residual blocks follow
        for _ in range(num_resd_block):
            x = self.residual_block(x)

        # Feed residual blocks output to the policy and value heads
        self.p_head = self.policy_head(x)
        self.v_head = self.value_head(x)

        # Create and compile the Keras model
        self.model = ks.Model(inputs=self.input_layer, outputs=[self.p_head, self.v_head])
        self.model.compile(**self.compile_parameters)

    def convolutional_block(self, x):
        """
        Create a convolutional block
        :param x: The input of the block
        :return: The output of the block
        """
        x = Conv2D(**self.conv_block_params,
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def residual_block(self, x):
        """
        Create a residual block
        :param x: The input of the block
        :return: The output of the block
        """
        x_in = x
        x = Conv2D(**self.resd_block_params,
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)  # TODO -- LeakyReLU?
        x = Conv2D(**self.resd_block_params,
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = add([x, x_in])
        x = Activation('relu')(x)
        return x

    def policy_head(self, x):
        """
        Create layers that form the policy output of the network
        :param x: The input of the policy head
        :return: The output policy
        """
        x = Conv2D(filters=2,
                   kernel_size=(1, 1),
                   strides=1,
                   padding='same',
                   kernel_regularizer=ks.regularizers.l2(self.c_L2reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(self.output_size, activation='softmax', name='policy')(x)
        return x

    def value_head(self, x):
        """
        Create layers that form the value output of the network
        :param x: The input of the value head
        :return: The output value
        """
        x = Conv2D(filters=1,
                   kernel_size=(1, 1),
                   strides=1,
                   padding='same',
                   kernel_regularizer=ks.regularizers.l2(self.c_L2reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)  # TODO -- proper params
        x = Dense(1, activation='tanh', name='value')(x)
        return x

    def get_model(self) -> ks.Model:
        """
        :return: The Keras model responsible for outputting the (p, v) tuples
        """
        return self.model

    def state_input(self, s: GameState):
        raise NotImplementedError

    def deepcopy(self):
        raise NotImplementedError
