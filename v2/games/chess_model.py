import numpy as np

from v2.model import ResidualNeuralNetwork
from v2.test_chess import Chess

from chess import PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING

PIECE_INDICES = {
    PAWN: 0,
    ROOK: 1,
    KNIGHT: 2,
    BISHOP: 3,
    QUEEN: 4,
    KING: 5
}

CHANNELS = len(PIECE_INDICES)

INPUT_SHAPE = Chess.board_shape() + (CHANNELS,)


class ChessModel(ResidualNeuralNetwork):

    def __init__(self):
        super().__init__(Chess, INPUT_SHAPE, num_resd_block=13)

    def state_input(self, s: Chess):
        m = np.zeros(shape=INPUT_SHAPE)
        for sq, p in s.board.piece_map().items():
            i = PIECE_INDICES[p.piece_type]
            x, y = Chess.square_index_to_xy(sq)
            m[x][y][i] = 1 if p.color == s.board.turn else -1
        return m

    def deepcopy(self):
        self.model.save_weights('checkpoint.pth.tar')
        self_copy = ChessModel()
        self_copy.model.load_weights('checkpoint.pth.tar')
        return self_copy


if __name__ == '__main__':
    from v2.self_play import policy_iter_self_play

    _m = ChessModel()

    _m = policy_iter_self_play(Chess,
                               _m,
                               num_iter=100,
                               num_exam=90000000,
                               num_sims_epis=100,
                               num_sims_duel=100,
                               num_duel=30,
                               num_expl=15,
                               num_epis=100)
