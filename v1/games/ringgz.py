from collections import Counter

import numpy as np
from v1.game import GameState

# Allowed number of player boundaries
MIN_NR_OF_PLAYERS, MAX_NR_OF_PLAYERS = 2, 4
# Board dimensions
MAX_X, MAX_Y = 5, 5
# Possible ring sizes
SIZES = 4
# Player colors
COLORS = ['blue', 'green', 'yellow', 'red']
# Start piece
START_PIECE = [('ring', COLORS[s], s) for s in range(SIZES)]
WAIT_TOKEN = ('wait',)
# Allowed start boundaries
MIN_START_X, MAX_START_X = 1, 3
MIN_START_Y, MAX_START_Y = 1, 3


class Ringgz(GameState):

    def __init__(self, number_of_players: int):
        assert MIN_NR_OF_PLAYERS <= number_of_players <= MAX_NR_OF_PLAYERS

        self.territories = [[None for _ in range(MAX_Y)] for _ in range(MAX_X)]

        # The total number of players in this game
        self.number_of_players = number_of_players

        # Index of the player whose turn it is
        self.current_player = 0

        # Store the last move performed on the board
        self.last_move = None

        # List all pieces for each player
        self.player_pieces = {p: [] for p in range(number_of_players)}

        # List the colors that each player can use to move
        self.player_colors = {p: [] for p in range(number_of_players)}

        # List the colors that each player can use to score points
        self.score_colors = {p: [] for p in range(number_of_players)}

        # Store if each player is able to move (game stopping criterion)
        self.able_to_move = {p: True for p in range(number_of_players)}

        # Store zobrist hash for this game state
        self.hash = 0

        self._init_player_pieces()
        self._init_player_colors()
        self._init_score_colors()

    def _init_player_pieces(self):
        if self.number_of_players == 2:
            for i in range(0, self.number_of_players * 2, 2):
                for j in range(3):
                    self.player_pieces[i // 2].append(('base', COLORS[i]))
                    self.player_pieces[i // 2].append(('base', COLORS[i + 1]))
                    for s in range(SIZES):
                        self.player_pieces[i // 2].append(('ring', COLORS[i], s))
                        self.player_pieces[i // 2].append(('ring', COLORS[i + 1], s))
        elif self.number_of_players == 3:
            for i in range(self.number_of_players):
                for j in range(3):
                    self.player_pieces[i].append(('base', COLORS[i]))
                    for s in range(SIZES):
                        self.player_pieces[i].append(('ring', COLORS[i], s))
                for s in range(SIZES):
                    self.player_pieces[i].append(('ring', COLORS[3], s))
        elif self.number_of_players == 4:
            for i in range(self.number_of_players):
                for j in range(3):
                    self.player_pieces[i].append(('base', COLORS[i]))
                    for s in range(SIZES):
                        self.player_pieces[i].append(('ring', COLORS[i], s))

    def _init_player_colors(self):
        if self.number_of_players == 2:
            for i in range(0, self.number_of_players * 2, 2):
                self.player_colors[i // 2].append(COLORS[i])
                self.player_colors[i // 2].append(COLORS[i + 1])
        elif self.number_of_players == 3:
            for i in range(self.number_of_players):
                self.player_colors[i].append(COLORS[i])
                self.player_colors[i].append(COLORS[3])
        elif self.number_of_players == 4:
            for i in range(self.number_of_players):
                self.player_colors[i].append(COLORS[i])

    def _init_score_colors(self):
        if self.number_of_players == 2:
            for i in range(0, self.number_of_players * 2, 2):
                self.score_colors[i // 2].append(COLORS[i])
                self.score_colors[i // 2].append(COLORS[i + 1])
        elif self.number_of_players == 3:
            for i in range(self.number_of_players):
                self.score_colors[i].append(COLORS[i])
        elif self.number_of_players == 4:
            for i in range(self.number_of_players):
                self.score_colors[i].append(COLORS[i])

    def __str__(self):
        string = ''
        for x in range(MAX_X):
            string += '|'
            for y in range(MAX_Y):
                territory = self.territories[x][y]
                if territory is None:
                    string += '    ,' * SIZES
                if isinstance(territory, tuple):
                    string += ' ' + territory[1][0] + territory[0][0] + ' ,'
                    string += '    ,' * (SIZES - 1)
                if isinstance(territory, list):
                    for size in range(SIZES):
                        if territory[size] is None:
                            string += '    ,'
                        else:
                            string += ' ' + territory[size][1][0] + territory[size][0][0] + str(territory[size][2]) + ','
                string += '|'
            string += '\n'
        return string

    def __hash__(self):
        return int(self.hash)

    @staticmethod
    def _get_type_of_piece(piece: tuple):
        return piece[0]

    @staticmethod
    def _get_color_of_piece(piece: tuple):
        return piece[1]

    @staticmethod
    def _get_size_of_piece(piece: tuple):
        return piece[2]

    @staticmethod
    def is_valid_territory(x: int, y: int):
        return 0 <= x < MAX_X and 0 <= y < MAX_Y

    @staticmethod
    def is_valid_starting_territory(x: int, y: int):
        return MIN_START_X <= x <= MAX_START_X and MIN_START_Y <= y <= MAX_START_Y

    def get_possible_moves(self):
        moves = []
        if self.last_move is None:
            for x in range(MIN_START_X, MAX_START_X + 1):
                for y in range(MIN_START_X, MAX_START_X + 1):
                    moves.append((x, y, None))
            return moves
        else:
            for x in range(MAX_X):
                for y in range(MAX_Y):
                    territory = self.territories[x][y]
                    if isinstance(territory, tuple):
                        # There is a base on this territory
                        pass
                    elif territory is None or territory == ([None] * SIZES):
                        # There is nothing on this territory
                        adj_colors, adj_base_colors = self._adjacent_colors(x, y)
                        for piece in self.player_pieces[self.current_player]:
                            if self._get_type_of_piece(piece) == 'base':
                                color = self._get_color_of_piece(piece)
                                if color in adj_colors and color not in adj_base_colors:
                                    moves.append((x, y, piece))
                            else:
                                if self._get_color_of_piece(piece) in adj_colors:
                                    moves.append((x, y, piece))
                    else:
                        # There are some rings on this territory
                        adj_colors, _ = self._adjacent_colors(x, y)
                        for piece in self.player_pieces[self.current_player]:
                            if self._get_type_of_piece(piece) == 'ring':
                                if self._get_color_of_piece(piece) in adj_colors:
                                    if territory[self._get_size_of_piece(piece)] is None:
                                        moves.append((x, y, piece))
            if len(moves) == 0:
                return [WAIT_TOKEN]
            return list(set(moves))

    def _adjacent_colors(self, x: int, y: int):
        colors, adj_base_colors = set(), set()
        for (x_, y_) in [(x + i, y + j) for i, j in [(0, 1), (1, 0), (-1, 0), (0, -1)]]:
            if self.is_valid_territory(x_, y_):
                nbh_territory = self.territories[x_][y_]
                if nbh_territory is None:
                    continue
                if isinstance(nbh_territory, tuple):
                    # Add the color of the base
                    colors.add(self._get_color_of_piece(nbh_territory))
                    adj_base_colors.add(self._get_color_of_piece(nbh_territory))
                else:
                    # Add the colors of all pieces
                    for piece in nbh_territory:
                        if isinstance(piece, tuple):
                            colors.add(self._get_color_of_piece(piece))
        return colors, adj_base_colors

    def is_terminal(self):
        return not any(self.able_to_move.values())

    def do_move(self, move: tuple):
        if move == WAIT_TOKEN:
            self.able_to_move[self.current_player] = False  # TODO -- proper
            self._switch_players()
            return self
        x, y, piece = move
        if piece is None:
            if self.is_valid_starting_territory(x, y):
                self._execute_move(x, y, START_PIECE, start=True)
            else:
                pass  # TODO crash
            return self
        else:
            territory = self.territories[x][y]
            if territory is None:
                if self._get_type_of_piece(piece) == 'ring':
                    self.territories[x][y] = [None] * SIZES
                self._execute_move(x, y, piece)
                return self
            elif isinstance(territory, tuple):
                # There's a base on this territory!
                pass  # TODO crash
                return self
            elif isinstance(territory, list):
                if territory[self._get_size_of_piece(piece)] is None:
                    self._execute_move(x, y, piece)
                else:
                    pass  # TODO crash
                return self

    '''
        Zobrist hashing
    '''
    _piece_indices = dict()
    i = 1
    for color in COLORS:
        _piece_indices[('base', color)] = i
        i += 1
    for color in COLORS:
        for size in range(SIZES):
            _piece_indices[('ring', color, size)] = i
            i += 1

    _zobrist_table = np.random.randint(1, high=(2 ** 31) - 1, size=(i, MAX_X, MAX_Y, SIZES))

    def _execute_move(self, x: int, y: int, piece, start=False):
        if start:
            self.territories[x][y] = START_PIECE
            self.last_move = (x, y, 'start')
            for i, p in enumerate(START_PIECE):
                self.hash ^= self._zobrist_table[self._piece_indices[p]][x][y][i]
        elif self._get_type_of_piece(piece) == 'base':
            self.territories[x][y] = piece
            self.last_move = (x, y, piece)
            self.hash ^= self._zobrist_table[self._piece_indices[piece]][x][y][0]
            self.player_pieces[self.current_player].remove(piece)
        elif self._get_type_of_piece(piece) == 'ring':
            size = self._get_size_of_piece(piece)
            self.territories[x][y][size] = piece
            self.last_move = (x, y, piece)
            self.hash ^= self._zobrist_table[self._piece_indices[piece]][x][y][size]
            self.player_pieces[self.current_player].remove(piece)
        self._switch_players()

    def _switch_players(self):
        self.current_player = (self.current_player + 1) % self.number_of_players
        return self.current_player

    def get_majorities(self):
        scores = [0] * self.number_of_players
        for x in range(MAX_X):
            for y in range(MAX_Y):
                territory = self.territories[x][y]
                if isinstance(territory, list):
                    colors = [self._get_color_of_piece(p) for p in territory if p is not None]
                    color_counts = Counter(colors)
                    highest, highest_color = -(2**31), None
                    for color, count in color_counts.items():
                        if count > highest:
                            highest, highest_color = count, color
                        elif count == highest:
                            highest_color = None
                    for player in range(self.number_of_players):
                        if highest_color in self.score_colors[player]:
                            scores[player] += 1
        return scores

    @staticmethod
    def get_all_actions():
        actions = [WAIT_TOKEN]
        for x in range(MIN_START_X, MAX_START_X + 1):
            for y in range(MIN_START_Y, MAX_START_Y + 1):
                actions.append((x, y, None))
        for x in range(MAX_X):
            for y in range(MAX_Y):
                for c in COLORS:
                    for s in range(SIZES):
                        actions.append((x, y, ('ring', c, s)))
                    actions.append((x, y, ('base', c)))
        return actions


if __name__ == '__main__':

    # state = Ringgz(4)
    # state.do_move((2, 2, None))
    #
    # print(state.get_possible_moves())
    #
    # for _ in range(100):
    #     moves = state.get_possible_moves()
    #     print(moves)
    #     if len(moves) == 0:
    #         state._switch_players()
    #         continue
    #     move = moves[np.random.choice(len(moves))]
    #     print(move)
    #     state.do_move(move)
    #
    # print(state.get_majorities())
    #
    # print(state)

    pass


