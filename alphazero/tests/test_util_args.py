import argparse


def get_dummy_args():
    parser = argparse.ArgumentParser(description='Dummy arguments for testing purposes')

    # Self play parameters
    parser.add_argument('--num_iter',
                        type=int,
                        default=10,
                        help='Number of iterations of self play during policy iteration. ' +
                             'Value in original paper: TODO')  # TODO
    parser.add_argument('--num_epis',
                        type=int,
                        default=10,
                        help='Number of episodes that are played to generate examples to train from in each iteration' +
                             ' of self play.' +
                             ' Value in original paper: 25000')
    parser.add_argument('--num_exam',
                        type=int,
                        default=10,
                        help='Number of most recent examples that are used for training a model. ' +
                             'Value in original paper: 500000')
    parser.add_argument('--win_threshold',
                        type=float,
                        default=0.55,
                        help='Threshold for the ratio of games that a challenging model has to win in order to be ' +
                             'considered better than the current model. ' +
                             'Value in the original paper: 0.55'
                        )
    parser.add_argument('--save_interval',
                        type=int,
                        default=10,
                        help='The number of iterations between saving the model to disk')

    # Episode parameters
    parser.add_argument('--num_sims_epis',
                        type=int,
                        default=10,
                        help='Number of Monte Carlo Tree searches that are executed before selecting a move during ' +
                             'the execution of an episode. ' +
                             'Value in original paper: 1600')
    parser.add_argument('--num_expl',
                        type=int,
                        default=3,
                        help='Number of initial exploring moves during the execution of an episode. That is, the ' +
                             'first NUM_EXPL moves are generated with tau=1 instead of tau=0. ' +
                             'Value in original paper: 30'
                        )

    # Duel parameters
    parser.add_argument('--num_sims_duel',
                        type=int,
                        default=10,
                        help='Number of Monte Carlo Tree searches that are executed before selecting a move when two ' +
                             'models are compared. ' +
                             'Value in original paper: 1600')
    parser.add_argument('--num_duels',
                        type=int,
                        default=10,
                        help='Number of duels that are played when two models are compared to see which is best. ' +
                             'Value in original paper: 400')

    # Model parameters
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='Learning rate when training the model using SGD ' +
                             'Value in original paper: 0.01 (But decreased with number of iterations)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='Momentum when training the model using SGD' +
                             'Value in original paper: 0.9')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size when training the model using SGD' +
                             'Value in original paper: 2048')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs executed when training the model on generated examples')
    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        help='Level of verbosity when training the model')
    parser.add_argument('--c_l2reg',
                        type=float,
                        default=1e-4,
                        help='' +
                             'Value in original paper: 1e-4')

    # Monte Carlo parameters
    parser.add_argument('--c_puct',
                        type=float,
                        default=1,
                        help='Constant determining the level of exploration during Monte Carlo Tree Search' +
                             'Value in original paper: TODO')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.03,
                        help='Parameterizes the Dirichlet distribution used to add noise to the initial moves to ' +
                             'increase exploration. ' +
                             'Value in original paper: 0.03')
    parser.add_argument('--epsilon',
                        type=float,
                        default=0.25,
                        help='Determines the strength of the noise that is added during the initial moves' +
                             'Value in original paper: 0.25')

    return parser.parse_args()