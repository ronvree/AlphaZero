
"""
    Number of iterations of self play during policy iteration
    Value in original paper: TODO
"""
NUM_ITER = 1000000

"""
    Number of episodes that are played to generate examples to train from in each iteration of self play
    Value in original paper: 25000
"""
NUM_EPIS = 100

"""
    Number of most recent examples that are used for training a model
    Value in original paper: 500000
"""
NUM_EXAM = 5000

"""
    Number of Monte Carlo Tree searches that are executed before selecting a move during the execution of an episode
    Value in original paper: 1600
"""
NUM_SIMS_EPIS = 100

"""
    Number of Monte Carlo Tree searches that are executed before selecting a move when two models are compared
    Value in original paper: 1600
"""
NUM_SIMS_DUEL = 100

"""
    Number of duels that are played when two models are compared to see which is best
    Value in original paper: 400
"""
NUM_DUEL = 30

"""
    Number of initial exploring moves during the execution of an episode. That is, the first NUM_EXPL moves are 
    generated with tau=1 instead of tau=0
    Value in original paper: 30
"""
NUM_EXPL = 30

"""
    Threshold for the ratio of games that a challenging model has to win in order to be considered better than
    the current model
    Value in the original paper: 0.55
"""
WIN_THRESHOLD = 0.55


