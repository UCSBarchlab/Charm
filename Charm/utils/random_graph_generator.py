import numpy as np

from Charm.interpreter.graph import *

""" This class generates a random dependency graph to test the conversion overhead of Charm.
"""

def getName(i):
    a = ord('a')
    name = ''
    while(True):
        name = str(chr(a + i % 7)) + name
        i = i / 7
        if i == 0:
            break
        i += -1
    return name

class DummyEq(object):
    def __init__(self, names):
        self.str = 'dummy'
        self.names = names

class RandomGraph(object):
    def __init__(self, M, N, sparse=False, distribute=True, drawable=False):
        """ Creates random dependency graph with M variables and N equations.
        """
        self.graph = Graph(drawable=drawable)
        names = [getName(i) for i in range(M)]
        name2node = {}
        for name in names:
            node = GraphNode(NodeType.VARIABLE, name)
            self.graph.addNode(node)
            name2node[name] = node
        count = [1] * len(names)
        eq_nodes = []
        for i in range(N):
            inv_prob = [1./c for c in count]
            tot = sum(inv_prob)
            inv_prob = [i / tot for i in inv_prob]
            dependency = np.random.choice(names,
                    size=np.random.randint(M / 2, M) if not sparse else np.random.randint(0, M / 2 + 1),
                    replace = False,
                    p = None if not distribute else inv_prob).tolist()
            for n in dependency:
                count[names.index(n)] += 2
            eq_node = GraphNode(NodeType.EQUATION, DummyEq(dependency))
            for n in dependency:
                edge = GraphEdge(eq_node, name2node[n])
                eq_node.addEdge(edge)
                name2node[n].addEdge(edge)
                self.graph.addEdge(edge)
            self.graph.addNode(eq_node)
            # Make sure graph is connected.
            eq_nodes = [n for n in self.graph.node_set if n.getType() == NodeType.EQUATION]
            for n in list(name2node.values()):
                for v in eq_nodes:
                    if not v is n:
                        if not self.graph.isConnected(n, v):
                            edge = GraphEdge(n, v)
                            v.addEdge(edge)
                            # Update dependency list.
                            v.val.names.append(n.val)
                            n.addEdge(edge)
                            self.graph.addEdge(edge)

    def getGraph(self):
        return self.graph

    def getNxGraph(self):
        G = nx.Graph()
        edges = []
        var_node = []
        rel_node = []
        for v in self.graph.node_set:
            if v.getType() == NodeType.VARIABLE:
                var_node.append(v.id)
                for e in v.edges:
                    edges.append((v.id, v.next(e).id))
            else:
                assert v.getType() == NodeType.EQUATION
                rel_node.append(v.id)
        G.add_nodes_from(var_node, bipartite=0)
        G.add_nodes_from(rel_node, bipartite=1)
        G.add_edges_from(edges)
        assert nx.is_connected(G)
        return G
