from copy import deepcopy

from v2.game import GameState
from v2.model import Model
from v2.montecarlo import MCST


class GameTree:
    """
        Game tree class for testing purposes

        A graph with Game States as nodes and actions as branches

        Each node has a dictionary that can store arbitrary attributes

        Edges can have attributes by storing a dictionary in the node that maps attributes over all
        outgoing edges

    """

    def __init__(self, s: GameState, depth: int, attr_func: callable = None):
        """
        Create a new game tree
        :param s: The root state of the tree
        :param depth: The depth of the tree (That is, the number of moves that will be looked ahead)
        :param attr_func: A function that, when given some state, outputs a dictionary of attributes that should
                          be stored in the game tree node corresponding to that state
        """
        self.node = {'layer': depth, 'state': s}

        # Add the attributes obtained from attr_func
        if attr_func is not None:
            attr = attr_func(s)
            self.node = {**self.node, **attr}

        if depth == 0:
            # This node is a leaf
            self.branches = {}
        else:
            # Expand the tree to the next layer
            self.branches = {a: GameTree(deepcopy(s).do_move(a),
                                         depth - 1,
                                         attr_func) for a in s.get_actions()}

    @staticmethod
    def from_mcst(s: GameState, depth: int, mcst: MCST):
        """
        Constructs a GameTree with attributes obtained from the given MCST
        :param s: The root state of the GameTree
        :param depth: The depth of the GameTree
        :param mcst: The Monte Carlo Search Tree that the attributes will be obtained from
        :return: a GameTree with extra attributes obtained from the MCST
        """

        # Define a function to obtain attributes from the MCST
        def get_mcst_attr(state: GameState, t: MCST):
            ks = MCST.key_transform(state)
            if ks in t.keys():
                return {'mcst_stats': t[ks]}
            else:
                return {'mcst_stats': None}

        # Pass the function to the GameTree constructor
        return GameTree(s, depth, lambda x: get_mcst_attr(x, mcst))

    @staticmethod
    def from_model(s: GameState, depth: int, model: Model):
        """
        Constructs a GameTree with the predictions of the model as attributes in each node
        :param s: The root state of the GameTree
        :param depth: The depth of the GameTree
        :param model: The model that will perform the (policy, value) predictions
        :return: a GameTree with extra attributes obtained from the model
        """

        # Define a function to obtain attributes from the model
        def get_model_attr(state: GameState, m: Model):
            o = state
            p, v = m.predict(state)
            return {'model_prediction': {
                'input': o,
                'p': p,
                'v': v
            }}

        # Pass the function to the GameTree Constructor
        return GameTree(s, depth, lambda x: get_model_attr(x, model))

    def __getitem__(self, item):
        """
        Get the subtree corresponding to taking the given action
        :param item: The action that should be taken
        :return: A GameTree that results from doing the action
        """
        return self.branches[item]

    def __iter__(self):
        """
        Iterate through the branches of this GameTree's root node
        """
        for branch in self.branches.values():
            yield branch

    def as_dot(self, graph_name: str = 'G', edge_val: callable = None) -> str:
        """
        Turn this game tree to a string in dot format
        :param graph_name: The name of the graph in the resulting dot text
        :param edge_val: Function that, when given dict of node attributes and an outgoing edge, gives a value that is
                         to be visualized in the tree
        :return: A string with this tree in dot
        """
        dot = 'digraph {} '.format(graph_name) + ' {\n'
        dot += 'graph [bgcolor=black, fontcolor=white, color=white];\n'
        dot += 'node [bgcolor=black, fontcolor=white, color=white];\n'
        dot += 'edge [fontcolor=white, color=white];\n'
        dot += self._as_dot(edge_val=edge_val)
        dot += '}\n'
        return dot

    def _as_dot(self, edge_val: callable = None) -> str:  # TODO -- special node visualization
        """
        Helper function of self.as_dot to be able to generate dot for subtrees recursively
        :return: Partial dot representation of this tree
        """
        state = self.node['state']

        # Hack the game's string representation into something suitable for GraphViz
        # TODO -- let games implement a functions that is always compatible with this tree (instead of using str)
        label = str(self.node['state']) \
            .replace('\n', '\\n') \
            .replace(' ', '   ') \
            .replace('-', '--')

        # Create a node with unique id (hash of state) and string representation as label
        dot = '{}[label=\"{}\",' \
              ' shape=box,' \
              ' bgcolor=black,' \
              ' fontcolor=white];\n'.format(hash(state), label)

        # Create the dot lines for each subtree
        for a, b in self.branches.items():
            # Check if edges require special visualization
            if edge_val is None:
                # Default edge visualization:
                dot += '{}->{};\n'.format(hash(state), hash(b.node['state']))
            else:
                # Edge visualization with focus on some edge attribute:
                attr = edge_val(a, self.node)
                sum_attr = sum([edge_val(aa, self.node) for aa in self.branches.keys()])
                if sum_attr == 0:
                    attr, sum_attr = 0, 1

                penwidth = int(1 + 10 * abs(attr / sum_attr))

                dot += '{}->{} [penwidth={}, label={}, fontcolor=white];\n' \
                    .format(hash(state), hash(b.node['state']), penwidth, attr)

            dot += b._as_dot(edge_val=edge_val)

        return dot


def get_mcts_edge_attr_p(a, node):
    """
    Function that gets the mcts p edge attribute from a dict of node attributes
    :param a: The action corresponding to the edge
    :param node: The node the attribute should be obtained from
    :return: the value of the attribute
    """
    return node['mcst_stats'][0][a]


def get_mcts_edge_attr_q(a, node):
    """
    Function that gets the mcts q edge attribute from a dict of node attributes
    :param a: The action corresponding to the edge
    :param node: The node the attribute should be obtained from
    :return: the value of the attribute
    """
    return node['mcst_stats'][2][a]


def get_mcts_edge_attr_n(a, node):
    """
    Function that gets the mcts n edge attribute from a dict of node attributes
    :param a: The action corresponding to the edge
    :param node: The node the attribute should be obtained from
    :return: the value of the attribute
    """
    return node['mcst_stats'][3][a]


def get_model_edge_attr_p(a, node):
    """
    Function that gets the model p edge attribute from a dict of node attributes
    :param a: The action corresponding to the edge
    :param node: The node the attribute should be obtained from
    :return: the value of the attribute
    """
    return node['model_prediction']['p'][a]


# TODO -- add special visualization for node values


if __name__ == '__main__':
    from v2.tictactoe import TicTacToe
    from v2.model import Model
    from v2.tictactoe_model import TicTacToeModel

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
