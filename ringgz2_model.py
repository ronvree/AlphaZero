

import keras as ks
from keras.layers import *
from keras.optimizers import SGD

from game import GameState
from games.ringgz import MAX_X, MAX_Y, SIZES
from games.ringgz_2 import Ringgz2
from model import Model


class TestNetwork(Model):

    def __init__(self):
        self.action_space = Ringgz2.get_all_actions()
        self.output_size = len(self.action_space)

        self.input_board = Input(shape=(MAX_X, MAX_Y, (SIZES + 1) * 2 * 2, 1), name='input')

        conv_layer1 = Activation('relu')(BatchNormalization()(Conv3D(128, 4, padding='same')(self.input_board)))
        conv_layer2 = Activation('relu')(BatchNormalization()(Conv3D(64, 3, padding='same')(conv_layer1)))
        conv_layer3 = Activation('relu')(BatchNormalization()(Conv3D(64, 3)(conv_layer2)))
        conv_layer4 = Activation('relu')(BatchNormalization()(Conv3D(64, 3)(conv_layer3)))
        conv_flat = Flatten()(conv_layer4)

        dropout_layer = Dropout(0.1)(Activation('relu')(BatchNormalization()(Dense(256)(conv_flat))))

        self.pi = Dense(self.output_size, activation='softmax', name='pi')(dropout_layer)
        self.v = Dense(1, activation='tanh', name='v')(dropout_layer)

        self.model = ks.Model(inputs=self.input_board, outputs=[self.pi, self.v])

        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=SGD(lr=0.001))

    def predict(self, s: GameState):
        state = s.get_observation()
        state = np.reshape(state, (1, MAX_X, MAX_Y, (SIZES + 1) * 2 * 2, 1))

        [pi_all], v = self.model.predict(state)

        legal_actions = s.get_possible_moves()
        mask = np.zeros(self.output_size)
        for a in legal_actions:
            mask[self.action_space.index(a)] = 1
        pi_all *= mask
        pi_all = pi_all.astype(np.float64)
        pi_all /= pi_all.sum()  # re-normalize probabilities

        pi = {a: pi_all[self.action_space.index(a)] for a in legal_actions}
        return pi, v

    def fit_new_model(self, examples):
        self.model.save_weights('checkpoint.pth.tar')
        self_copy = TestNetwork()
        self_copy.model.load_weights('checkpoint.pth.tar')

        # Prepare all examples for training
        input_boards, target_pis, target_vs = [], [], []
        for s, pi, v in examples:
            # Reshape the state observation to an input suitable for convolution
            input_boards.append(np.reshape(s, (MAX_X, MAX_Y, (SIZES + 1) * 2 * 2, 1)))
            # # Map all actions to a fixed index
            pi_ = np.zeros(self.output_size)
            for a, p in pi.items():
                pi_[self.action_space.index(a)] = p
            target_pis.append(pi_)
            # Add reward value
            target_vs.append(v)

        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        self.model.fit(x=input_boards, y=[target_pis, target_vs], epochs=10, batch_size=128, shuffle=True, verbose=2)

        return self_copy
