from copy import deepcopy

from v2.game import GameState


WIN_SCORE = 10000


class MiniMaxAgent:

    def __init__(self, depth: int, color: int):
        self.depth = depth
        self.color = color
        self.best_move = None

    def select_action(self, s: GameState):
        self.best_move = None
        self.negamax(deepcopy(s), self.depth, self.color)
        return self.best_move

    def negamax(self, s: GameState, depth: int, color: int):
        if depth == 0 or s.is_terminal():
            return self.score(s)
        actions = self.order_moves(s.get_actions())
        a, v = max([(a, -self.negamax(deepcopy(s).do_move(a), depth - 1, color * -1)) for a in actions], key=lambda x: x[1])
        if depth == self.depth:
            self.best_move = a
        return v

    def order_moves(self, moves: list) -> list:
        return moves

    def score(self, s: GameState):
        if s.is_terminal():
            r = s.get_reward() * WIN_SCORE
        else:
            r = self.heuristic(s)

        if self.color != s.get_current_player():
            r *= -1
        return r

    def heuristic(self, s: GameState):
        raise NotImplementedError


class MiniMaxAlphaBetaAgent(MiniMaxAgent):

    def negamax(self, s: GameState, depth: int, color: int, alpha: int = -WIN_SCORE, beta: int = WIN_SCORE):
        if depth == 0 or s.is_terminal():
            return self.score(s)
        actions = self.order_moves(s.get_actions())

        max_v = -WIN_SCORE - 1
        for a in actions:
            sp = deepcopy(s).do_move(a)
            v = -self.negamax(sp, depth - 1, color * -1, -beta, -alpha)
            if v >= max_v:
                max_v = v
                if self.depth == depth:
                    self.best_move = a
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return max_v

    def heuristic(self, s: GameState):
        raise NotImplementedError


if __name__ == '__main__':
    from v2.games.tictactoe import TicTacToe
    import random

    _s = TicTacToe()
    _s.do_move((2, 2))
    _s.do_move((2, 1))
    _s.do_move((2, 0))
    _s.do_move((1, 1))

    print(_s)

    _m = MiniMaxAlphaBetaAgent(9, 1)

    _m.heuristic = lambda s: random.randint(0, 10)

    _a = _m.select_action(_s)

    print(_a)


