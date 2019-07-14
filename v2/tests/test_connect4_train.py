

if __name__ == '__main__':
    from v2.games.connect4 import Connect4
    from v2.games.connect4_model import Connect4Model
    from v2.self_play import policy_iter_self_play

    _m = Connect4Model()

    # _m.model.load_weights('checkpoint.pth.tar')

    _m = policy_iter_self_play(Connect4,
                               _m,
                               num_iter=100,
                               num_exam=100000,
                               num_sims_epis=40,
                               num_sims_duel=40,
                               num_duel=30,
                               num_expl=15,
                               num_epis=100)
