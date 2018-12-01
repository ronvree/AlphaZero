from v1.model import Model


class Episode:

    def __init__(self, game: callable, model: Model):
        self.game = game
        self.model = model
        self.examples = []


        pass  # TODO

    def execute(self):
        # examples = []
        # mcts = MonteCarloSearchTree()
        # # Initialize the game state
        # state = game()
        # # Continue until the game ends
        # move_counter = 0
        # while not state.is_terminal():
        #     # Perform a number of Monte Carlo Tree Search simulations
        #     for _ in range(number_of_mcts_simulations):
        #         mcts.search(state, model)
        #
        #     # The first moves have a temperature of 1 for increased exploration
        #     tau = 1 if move_counter < number_of_exploring_moves else 0
        #
        #     # Obtain an improved policy from the tree
        #     a, pi = mcts.action(state, temperature=tau)
        #
        #     # Store examples for the model to train on. One example consists of:
        #     # - The game state from the current player's perspective
        #     # - A policy obtained from the search tree
        #     # - A value indicating if the player eventually won the game (so is added later)
        #     examples.append([state.get_observation(), pi])
        #
        #     # Perform the move
        #     state = state.do_move(a)
        #     move_counter += 1
        #
        # # If the game ended, add the final result to the examples and return them
        # # reward=1 if player 1 won, reward=-1 if player 2 won
        # reward = state.get_score()
        # for e in examples:
        #     e.append(reward)
        #     reward *= -1  # Switch player perspective
        # return examples
        pass  # TODO

    def get_examples(self):
        pass  # TODO

    def describe_game(self):
        pass  # TODO

    def describe_game_end(self):
        pass  # TODO

# def execute_episode(game: callable, model: Model):
#     """
#     Execute one episode of self-play
#     :param game: The game to be played
#     :param model: The model used to generate examples
#     :return: a list of examples (explained below) that is obtained during self-play
#     """
#     examples = []
#     mcts = MonteCarloSearchTree()
#     # Initialize the game state
#     state = game()
#     # Continue until the game ends
#     move_counter = 0
#     while not state.is_terminal():
#         # Perform a number of Monte Carlo Tree Search simulations
#         for _ in range(number_of_mcts_simulations):
#             mcts.search(state, model)
#
#         # The first moves have a temperature of 1 for increased exploration
#         tau = 1 if move_counter < number_of_exploring_moves else 0
#
#         # Obtain an improved policy from the tree
#         a, pi = mcts.action(state, temperature=tau)
#
#         # Store examples for the model to train on. One example consists of:
#         # - The game state from the current player's perspective
#         # - A policy obtained from the search tree
#         # - A value indicating if the player eventually won the game (so is added later)
#         examples.append([state.get_observation(), pi])
#
#         # Perform the move
#         state = state.do_move(a)
#         move_counter += 1
#
#     # If the game ended, add the final result to the examples and return them
#     # reward=1 if player 1 won, reward=-1 if player 2 won
#     reward = state.get_score()
#     for e in examples:
#         e.append(reward)
#         reward *= -1  # Switch player perspective
#     return examples

