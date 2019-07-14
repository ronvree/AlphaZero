from v2.games.tictactoe_model import TicTacToeModel

from v2.tests.self_play_simple import policy_iter_self_play


if __name__ == '__main__':
    from v2.games.tictactoe import TicTacToe

    _m = TicTacToeModel()

    # _m.model.load_weights('checkpoint.pth.tar')

    _m = policy_iter_self_play(TicTacToe,
                               _m,
                               num_iter=100,
                               num_exam=90000,
                               num_sims=100,
                               num_duel=30,
                               num_expl=3,
                               num_epis=100)
