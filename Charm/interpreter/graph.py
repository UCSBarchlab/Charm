""" Core graph stuctures of Charm.
"""

from collections import deque
from copy import copy, deepcopy

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from .abstract_syntax_tree import *


# A lattice for node type:
#   Top
#   / \
# CON INP
#  |   |
# EQA VAR
#   \ /
#   BOT
class NodeType(object):
    EQUATION = 0
    CONSTRAINT = 1
    VARIABLE = 2
    INPUT = 3
    INVALID = 4

    __printables = ['EQ', 'CON', 'VAR', 'INP']

    @staticmethod
    def getPrintable(nt):
        assert nt < NodeType.INVALID
        return NodeType.__printables[nt]


class GraphEdge(IdObject):
    """ An edge can be either directed or undirected.

    A directed edge should be refered using src, dst, while an
    undirected edge should be refered using node1, node2.
    """

    def __init__(self, node1, node2, src=None, dst=None):
        super(GraphEdge, self).__init__()
        self.fixed = False
        self.node1 = node1
        self.node2 = node2
        self.src = src
        self.dst = dst
        # If src and dst are both provided, this is a directed edge.
        if src and dst:
            assert (src is node1 and dst is node2) or (src is node2 and dst is node1)
        else:
            assert not src and not dst, \
                'Bad graph edge ({}->{})'.format(self.src.id, self.dst.id)

    def set(self, src, dst, fixed=False):
        # If an edge is fixed, it cannot be set/reset.
        assert not self.fixed, 'Trying to set fixed edge {}->{}'.format(self.src.id, self.dst.id)
        self.src = src
        self.dst = dst
        self.fixed = fixed

    def reset(self):
        assert not self.fixed, 'Trying to reset fixed edge {}->{}'.format(self.src.id, self.dst.id)
        self.src = self.dst = None

    def isDirected(self):
        if self.src and self.dst:
            return True
        elif not self.src and not self.dst:
            return False
        else:
            raise ValueError('Bad graph edge ({}->{})'.format(self.src.id, self.dst.id))

    def getPrintable(self):
        if not self.isDirected():
            return '{}{}{}'.format(self.node1.id, '--', self.node2.id)
        else:
            return '{}{}{}'.format(self.src.id, '->', self.dst.id)

    def __eq__(self, rhs):
        """ Equality exists when the nodes are the same, no matter directed or not.
        """
        return (self.node1 is rhs.node1 and self.node2 is rhs.node2) or \
               (self.node1 is rhs.node2 and self.node2 is rhs.node1)


class GraphNode(Node):
    def __init__(self, node_type, val):
        super(GraphNode, self).__init__()
        self.type = node_type
        self.vector = False
        # The actual value of the node, depending on the node type, it can be either:
        # 1. a variable name in string, or
        # 2. an relation node in the AST
        self.val = val
        # Set of name extensions on the node.
        self.exts = []
        # Set of edges connecting the node.
        self.edges = []
        self.ordered_given = []
        self.proped = {}
        self.marked = False
        self.func = None
        self.func_str = None
        # Output variable name.
        self.out_name = None
        # Output variable value.
        self.out_val = None

    def getPrintable(self):
        return '{} ({})\n\t{}'.format(self.val if (self.getType() == NodeType.INPUT or
                                                   self.getType() == NodeType.VARIABLE) else self.val.str,
                                      NodeType.getPrintable(self.getType()),
                                      '\n\t'.join([e.getPrintable() for e in self.edges]))

    def dump(self, **kwargs):
        logging.info('GraphNode {} {}:'.format(self.id, 'marked' if self.marked else ''))
        logging.info('\t{}'.format(self.getPrintable()))
        logging.info('\tgiven: {}'.format(self.ordered_given))
        logging.info('\tproped: {}'.format(self.proped))

    def clone(self, ext, ext_set):
        """ Make a clone of the current node with ext, given the entire ext_set in the graph.
        """
        node = GraphNode(self.type, None)
        node.marked = self.marked
        node.vector = self.vector
        node.exts = deepcopy(self.exts)
        node.addExtName(ext)
        for e in self.edges:
            nb = self.next(e)
            if e.isDirected():
                if e.src is self:
                    new_e = GraphEdge(node, nb, node, nb)
                    node.addEdge(new_e)
                    nb.addEdge(new_e)
                else:
                    assert e.dst is self
                    new_e = GraphEdge(nb, node, nb, node)
                    node.addEdge(new_e)
                    nb.addEdge(new_e)
            else:
                new_e = GraphEdge(node, nb)
                node.addEdge(new_e)
                nb.addEdge(new_e)

        if self.type == NodeType.VARIABLE:
            node.val = self.val + Names.clone_ext + ext
        elif self.type == NodeType.EQUATION or self.type == NodeType.CONSTRAINT:
            node.val = deepcopy(self.val)
            ext_names = []
            for e in node.edges:
                nn = node.next(e)
                if nn.getType() == NodeType.VARIABLE and nn.exts[-1] == ext:
                    ext_names.append(nn.val)
            dummy_edges = [e for e in node.edges if (node.next(e).getType() == NodeType.VARIABLE and
                                                     node.next(e).exts[-1] != ext) or (
                                   node.next(e).getType() == NodeType.INPUT and
                                   (set(node.next(e).exts) & ext_set) and
                                   (set(node.next(e).exts) & ext_set != ext))]
            for e in dummy_edges:
                nn = node.next(e)
                node.removeEdge(e)
                nn.removeEdge(e)
            logging.debug('Create clone {} from {} with {}'.format(node.id, self.id, ext_names))
            for ext_name in ext_names:
                node.val.subs(ext_name)
        else:
            raise ValueError('Should not clone input nodes.')
        return node

    def addExtName(self, ext):
        assert not ext in self.exts
        self.exts.append(ext)

    def setOutName(self, ostr=None):
        """ Constraint node does not output values, thus None.
        """
        self.out_name = ostr

    def addEdge(self, edge):
        for e in self.edges:
            assert not e == edge, 'Edge {} already exits'.format(e.getPrintable())
        self.edges.append(edge)

    def removeEdge(self, edge):
        assert edge in self.edges
        self.edges.remove(edge)

    def getFlexibleEdges(self):
        flexible = []
        for e in self.edges:
            if not e.node1.marked and not e.node2.marked:
                flexible.append(e)
        return flexible

    def next(self, edge):
        # Given an edge, get the neighbouring node, no matther directed or not.
        return edge.node1 if self is edge.node2 else edge.node2

    def mark(self):
        assert not self.marked
        self.marked = True

    def unmark(self):
        assert self.marked
        self.marked = False

    def setType(self, t):
        if self.type == NodeType.VARIABLE:
            assert t == NodeType.INPUT
            self.type = t
        elif self.type == NodeType.EQUATION:
            assert t == NodeType.CONSTRAINT
            self.type = t
        else:
            raise ValueError('Cannot set node type from {} to {}'.format(self.type, t))(self.getPrintable())

    def getType(self):
        return self.type

    def has_conflict(self):
        if self.type == NodeType.VARIABLE:
            indegree = 0
            for e in self.edges:
                if e.isDirected() and e.dst is self:
                    indegree += 1
            if indegree != 1:
                return True, indegree
        elif self.type == NodeType.EQUATION:
            outdegree = 0
            for e in self.edges:
                if e.isDirected() and e.src is self:
                    outdegree += 1
            if outdegree > 1:
                return True, outdegree
        elif self.type == NodeType.CONSTRAINT:
            outdegree = 0
            for e in self.edges:
                if e.isDirected() and e.src is self:
                    outdegree += 1
            if outdegree > 0:
                return True, outdegree
        else:
            assert self.type == NodeType.INPUT
            indegree = 0
            for e in self.edges:
                if e.isDirected() and e.dst is self:
                    indegree += 1
            if indegree > 0:
                return True, indegree
        return False, None


class Graph(object):
    def __init__(self, drawable=False):
        self.node_set = []
        self.edge_set = []
        self.drawable = drawable

    def addNode(self, node):
        assert node not in self.node_set
        self.node_set.append(node)

    def removeNode(self, node):
        assert node in self.node_set
        edges = copy(node.edges)
        for e in edges:
            self.removeEdge(e)
        self.node_set.remove(node)

    def addEdge(self, edge):
        assert edge not in self.edge_set
        self.edge_set.append(edge)

    def removeEdge(self, edge):
        """ When removing edge, remove from both endpoints.
        """
        assert edge in self.edge_set
        edge.node1.removeEdge(edge)
        edge.node2.removeEdge(edge)
        self.edge_set.remove(edge)

    def addEdges(self, edges):
        for edge in edges:
            assert not edge in self.edge_set
            self.edge_set.append(edge)

    def getUnmarkedNode(self):
        for n in self.node_set:
            if not n.marked:
                yield n

    def getNextEqNode(self):
        for n in self.node_set:
            if n.getType() == NodeType.EQUATION:
                yield n

    def getNextConNode(self):
        for n in self.node_set:
            if n.getType() == NodeType.CONSTRAINT:
                yield n

    def getNextRelationNode(self):
        for n in self.node_set:
            if n.getType() == NodeType.CONSTRAINT or n.getType() == NodeType.EQUATION:
                yield n

    def connected(self, src, dst):
        """ If connected in a directed graph.
        """
        nodes = deque()
        nodes.append(src)
        idx = 0
        while idx < len(nodes):
            cur = nodes[idx]
            if src is dst:
                return True
            for e in cur.edges:
                if not cur.next(e) in nodes:
                    nodes.append(cur.next(e))
            idx += 1
        return False

    def isConnected(self, node1, node2):
        connected = set()
        q = deque()
        q.append(node1)
        connected.add(node1)
        while q:
            cur = q.popleft()
            for e in cur.edges:
                nb = cur.next(e)
                if not nb in connected:
                    q.append(nb)
                connected.add(nb)
        return node2 in connected

    def hasPath(self, src, dst):
        """ If has a path in directed graph.
        """
        if src is dst:
            return True
        for e in src.edges:
            assert e.isDirected()
            if e.src == src:
                if self.hasPath(e.dst, dst):
                    return True
        return False

    def eval_constraints(self):
        """ Evaluate all constraint nodes.
        """
        for n in [node for node in self.node_set if node.getType() == NodeType.CONSTRAINT]:
            n.proped = {}
            for e in n.edges:
                assert e.isDirected()
                assert e.dst is n
                assert e.src.out_name in n.ordered_given
                if not e.src.out_val is None:
                    n.proped[e.src.out_name] = e.src.out_val
            if set(n.proped.keys()) == set(n.ordered_given):
                logging.debug('Eval Con with {}\nResult: {}'.format(
                    n.proped, n.func(**n.proped)))
                pass

            if set(n.proped.keys()) < set(n.ordered_given) or not n.func(**(n.proped)):
                logging.log(logging.ERROR, 'VIOLATION: [{}] on:\n\t{}'.format(n.val.str, n.proped))
                return False
        return True

    def check(self):
        """ Check for node violations in a functional graph.
        """
        for e in self.edge_set:
            assert e.isDirected(), \
                'Still has unlabeled edge {}->{}'.format(e.node1, e.node2)

        for n in self.node_set:
            has_conflict = n.has_conflict()[0]
            if has_conflict:
                logging.log(logging.FATAL, 'Conflict detected at node:\n{}'.format(n.getPrintable()))
            assert not has_conflict

    def draw(self, path='DAG', show=False):
        if not self.drawable:
            return

        logging.debug('Drawing {}'.format(path))
        plt.figure(figsize=(20, 10))
        G = nx.DiGraph()
        labels = {}
        colors = []
        for n in self.node_set:
            if n.getType() == NodeType.VARIABLE or n.getType() == NodeType.INPUT:
                logging.debug('Node {}: {} -> {}'.format(n.id, n.val, n.out_name))
            else:
                logging.debug('Node {}: {} -> {}'.format(n.id, n.val.str, n.out_name))
            G.add_node(n)
            labels[n] = n.id
        for e in self.edge_set:
            if e.isDirected():
                logging.debug('D Edge {}->{}'.format(e.src.id, e.dst.id))
                G.add_edge(e.src, e.dst)
            else:
                logging.debug('U Edge {}--{}'.format(e.node1.id, e.node2.id))
                G.add_edge(e.node1, e.node2)
        pos = graphviz_layout(G, prog='dot')
        nx.draw_networkx(G, pos, arrows=True, with_labels=False, alpha=.5,
                         node_color=[node.type for node in G.nodes()])
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        plt.axis('off')
        plt.savefig(path, dpi=200) if not show else plt.show()
        logging.info('Drawing saved to {}'.format(path))
