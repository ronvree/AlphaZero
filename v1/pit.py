from v1.model import Model


class Pit:

    def __init__(self, game: callable, m1: Model, m2: Model):
        self.game = game
        self.models = [m1, m2]

        pass  # TODO

    def __str__(self):
        pass  # TODO

    def duel(self):
        return self  # TODO

    def get_result(self):
        pass  # TODO

    def describe_games(self):
        pass  # TODO

    def describe_game_ends(self):
        pass  # TODO


# def pit(m_alt: Model, m_orig: Model, game: callable):
#     """
#     Let two models play a certain number of games against each other
#     :param m_alt: The 'alternative' model
#     :param m_orig: The 'original' model
#     :param game: The game that should be played
#     :return: the fraction of games won by the 'alternative' model
#     """
#     # Keep track of the number of games won for each model
#     wins = np.zeros(2)
#
#     # Initialize a search tree for both models
#     mcts_alt, mcts_orig = MonteCarloSearchTree(), MonteCarloSearchTree()  # TODO -- reset mcts each game? Stochasticity in games?
#
#     # Let the models play a number of games
#     for duel in range(number_of_duels):
#         # Store which model corresponds to which player
#         # Let the models take turns in who is the starting player
#         models = {duel % 2: (m_alt, mcts_alt), (duel + 1) % 2: (m_orig, mcts_orig)}
#
#         # Initialize a new game and play it
#         state = game()
#         current_player = 0
#         while not state.is_terminal():
#             model, tree = models[current_player]
#             for _ in range(number_of_mcts_simulations):
#                 tree.search(state, model)
#             # Determine an action by sampling from the policy as defined by the tree
#             a, _ = tree.action(state, temperature=0)  # Temperature = 0 for best move selection (no exploration)
#             state.do_move(a)
#             current_player = 1 - current_player
#         # Add the game result to the win counter (taking player perspective into account)
#         if duel % 2 == 0:
#             wins += state.get_scores()
#         else:
#             wins += np.roll(state.get_scores(), 1)
#     # Return the fraction of games won by the alternative model
#     if sum(wins) == 0:
#         return 0.5
#     else:
#         return wins[0] / sum(wins)
