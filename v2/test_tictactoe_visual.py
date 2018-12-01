

if __name__ == '__main__':
    from v2.tictactoe import TicTacToe
    from v2.tictactoe_model import TicTacToeModel
    from v2.montecarlo import MCST
    from v2.test_util_gametree import GameTree, get_model_edge_attr_p

    _s = TicTacToe()
    _m = TicTacToeModel()

    _m.get_model().load_weights('checkpoint.pth.tar')

    _mcst = MCST()

    for _ in range(2000):
        _mcst.search(_s, _m)

    _t = GameTree.from_model(_s, 3, _m)
    _dot = _t[0, 0].as_dot(edge_val=get_model_edge_attr_p)  # Probability distribution p output by model

    # _t = GameTree.from_mcst(_s, 4, _mcst)
    # _dot = _t[0, 0][1, 1].as_dot(edge_val=get_mcts_edge_attr_p)  # MCST Probability distribution pi
    # _dot = _t[0, 0][1, 1].as_dot(edge_val=get_mcts_edge_attr_q) # MCST Expected reward
    # _dot = _t[0, 0][1, 1].as_dot(edge_val=get_mcts_edge_attr_n) # MCST Visit count

    with open('test2.dot', 'w') as _f:
        _f.write(_dot)

