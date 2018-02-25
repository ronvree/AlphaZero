from connect4_model import TestNetwork
from games.connect4 import Connect4
from mcts import MonteCarloSearchTree

if __name__ == '__main__':
    import numpy as np

    game_state = Connect4()

    # Initialize network f
    checkpoint_model = TestNetwork()
    checkpoint_model.model.load_weights('checkpoint.pth.tar')
    # Initialize search tree alpha
    mcts = MonteCarloSearchTree()

    def determine_move(state):
        for _ in range(25):
            mcts.search(state, checkpoint_model)
        return mcts.action(state, temperature=1)[0]

    player1 = determine_move
    player2 = lambda s: int(input())
    player3 = lambda s: np.random.randint(7)

    players = [player1, player2]

    current = 0
    print(game_state)
    while not game_state.is_terminal():
        game_state.do_move(players[current](game_state))
        current = 1 - current
        print(game_state)
        print(game_state.winner)
