import functools
import io
import itertools
import pickle
from collections import defaultdict
from timeit import default_timer as timer

import mcerp3 as mcerp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from networkx.algorithms import bipartite as biGraph
from sympy import simplify
from sympy.parsing.sympy_parser import _token_splittable
from sympy.utilities.lambdify import lambdify, lambdastr

from Charm.base.helpers import *
from Charm.models import Dummy
from Charm.utils.random_graph_generator import RandomGraph
from .graph import *
from .smt_wrapper import SMTInstance
from .z3core import graph_transform_z3


def hasExtName(name):
    # A valid extended variable name can only have one dot extension.
    assert name.count(Names.clone_ext) <= 1, \
        'Mulitple extensions found in name {}'.format(name)
    return Names.clone_ext in name


def concatExtName(base, ext):
    return base + Names.clone_ext + ext


def getExtName(name):
    assert hasExtName(name)
    return name.split(Names.clone_ext)[1]


def getBaseName(name):
    if hasExtName(name):
        name = name.split(Names.clone_ext)[0]
    return name


class Interpreter(object):
    def __init__(self, program, use_z3=False, drawable=False, mcsamples=100):
        # This sets the sample number for MC.
        mcerp.npts = mcsamples
        self.use_z3 = use_z3
        self.drawable = drawable
        self.program = program
        self.nodes = program.nodes
        self.graph = Graph(self.drawable)
        self.var_set = set()  # Names of variables assumed.
        self.eq_set = set()  # Equations assumed.
        self.con_set = set()  # Constraints assumed.
        self.targets = []  # Target variables to solve for.
        self.v2t = {}  # Variable -> typeNode, assumed.
        self.given = {}  # Variables  -> value given in let stmt.
        self.assumptions = {}  # Variables -> value given in rule def.
        self.dependent = {}  # Variables -> dependent variables in let stmt.
        self.n2r = {}  # Rule name to RuleNode.
        self.v2v = {}  # Variable name to VarNode.
        self.n2t = {}  # Type name to TypeNode.
        self.v2l = {}  # Variable name to LetNode.
        self.plot_nodes = []  # Plot tasks
        self.stack = deque()

    def type_check(self, var, ival):
        """ Checks constraints associated with type.
        """
        if ival is None:
            return False
        if not isinstance(var, tuple) and not isinstance(ival, tuple):
            var = (var,)
            ival = (ival,)
        assert isinstance(var, tuple) and isinstance(ival, tuple), \
            'Inconsistent iterable type {}: {}'.format(var, ival)
        for var, v in zip(var, ival):
            var = getBaseName(var)
            if not isinstance(v, list):
                v = [v]
            for val in v:
                """
                # Data type checking.
                    typeNode = self.v2t[var]
                    type_val = type(val)
                    if type_val.__module__ == np.__name__:
                        type_val = type(np.asscalar(val))
                    assert type_val is eval(typeNode.data_type), \
                            'Type of {} is {}, but defined as {}'.format(
                                    var, type_val, typeNode.name)
                """
                typeNode = self.v2t[var]
                for con in typeNode.constraints:
                    expr = parse_expr(con)
                    if isinstance(val, mcerp.UncertainFunction):
                        value = val._mcpts
                        valid_p = []
                        for check_v in value:
                            if expr.subs(typeNode.use, check_v):
                                valid_p.append(check_v)
                        if valid_p:
                            val._mcpts = np.asarray(valid_p)
                            return True
                        else:
                            return False
                    else:
                        return expr.subs(typeNode.use, val)

    def DFS(self, cur):
        if cur.marked:
            logging.debug('[{}]: {} marked'.format(cur.id, cur.id))
            return True

        logging.debug('[{}]: push {} ({} edges)'.format(cur.id, cur.id, len(cur.edges)))
        assert cur.id not in self.stack
        self.stack.append(cur.id)
        logging.debug('stack: {}'.format(self.stack))

        indegree = outdegree = 0
        # First, get current indegree and outdegree.
        for e in cur.edges:
            if e.isDirected():
                if e.src is cur and e.dst.marked:
                    outdegree += 1
                    logging.debug('[{}]: {} {} {}'.format(
                        cur.id, cur.id, '->' if not e.fixed else '=>', e.dst.id))
                elif e.dst is cur and e.src.marked:
                    assert e.dst is cur
                    indegree += 1
                    logging.debug('[{}]: {} {} {}'.format(
                        cur.id, e.src.id, '->' if not e.fixed else '=>', cur.id))
                else:
                    continue

        flexibles = cur.getFlexibleEdges()

        cur.mark()

        if not flexibles:
            if not cur.has_conflict()[0]:
                logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                assert self.stack.pop() == cur.id
                cur.unmark()
                return True
            else:
                logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                assert self.stack.pop() == cur.id
                cur.unmark()
                return False

        if cur.getType() == NodeType.INPUT or cur.getType() == NodeType.CONSTRAINT:
            raise ValueError('Input/Costraint node {} seen during DFS'.format(cur.id))
        elif cur.getType() == NodeType.VARIABLE:
            if indegree == 1:
                logging.debug('[{}]: {} VAR indegree = 1'.format(cur.id, cur.id))
                for e in flexibles:
                    e.set(cur, cur.next(e))
                    logging.debug('[{}]: setting {}->{}'.format(cur.id, e.src.id, e.dst.id))
                all_marked = True
                for e in cur.edges:
                    logging.debug('[{}]: {} --> {}'.format(cur.id, cur.id, cur.next(e).id))
                    all_marked = all_marked & self.DFS(cur.next(e))
                    if not all_marked:
                        break;
                if all_marked:
                    logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                    assert self.stack.pop() == cur.id
                    cur.unmark()
                    return True
                else:
                    for e in flexibles:
                        if not cur.next(e).marked:
                            logging.debug('[{}]: resetting {}->{}'.format(cur.id, e.src.id, e.dst.id))
                            e.reset()
                    logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                    assert self.stack.pop() == cur.id
                    cur.unmark()
                    return False
            elif indegree == 0:
                logging.debug('[{}]: {} VAR indegree = 0'.format(cur.id, cur.id))
                for in_e in flexibles:
                    in_e.set(cur.next(in_e), cur)
                    logging.debug('[{}]: setting {}->{}'.format(cur.id, in_e.src.id, in_e.dst.id))
                    for e in flexibles:
                        if not e == in_e:
                            e.set(cur, cur.next(e))
                            logging.debug('[{}]: setting {}->{}'.format(cur.id, e.src.id, e.dst.id))
                    all_marked = True
                    for e in cur.edges:
                        logging.debug('[{}]: {} --> {}'.format(cur.id, cur.id, cur.next(e).id))
                        all_marked = all_marked & self.DFS(cur.next(e))
                        if not all_marked:
                            break;
                    if all_marked:
                        logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                        assert self.stack.pop() == cur.id
                        cur.unmark()
                        return True
                    else:
                        for e in flexibles:
                            if not cur.next(e).marked:
                                logging.debug('[{}]: resetting {}->{}'.format(
                                    cur.id, e.src.id, e.dst.id))
                                e.reset()
                logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                assert self.stack.pop() == cur.id
                cur.unmark()
                return False
            else:
                logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                assert self.stack.pop() == cur.id
                cur.unmark()
                return False
        else:
            assert cur.getType() == NodeType.EQUATION
            if outdegree == 1:
                logging.debug('[{}]: {} outdegree = 1'.format(cur.id, cur.id))
                for e in flexibles:
                    e.set(cur.next(e), cur)
                    logging.debug('[{}]: setting {}->{}'.format(cur.id, e.src.id, e.dst.id))
                all_marked = True
                for e in cur.edges:
                    logging.debug('[{}]: {} --> {}'.format(cur.id, cur.id, cur.next(e).id))
                    all_marked = all_marked & self.DFS(cur.next(e))
                    if not all_marked:
                        break
                if all_marked:
                    logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                    assert self.stack.pop() == cur.id
                    cur.unmark()
                    return True
                else:
                    for e in flexibles:
                        if not cur.next(e).marked:
                            logging.debug('[{}]: resetting {}->{}'.format(cur.id, e.src.id, e.dst.id))
                            e.reset()
                    logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                    assert self.stack.pop() == cur.id
                    cur.unmark()
                    return False
            elif outdegree == 0:
                logging.debug('[{}]: {} outdegree = 0'.format(cur.id, cur.id))
                for out_e in flexibles:
                    out_e.set(cur, cur.next(out_e))
                    logging.debug('[{}]: setting {}->{}'.format(cur.id, out_e.src.id, out_e.dst.id))
                    for e in flexibles:
                        if not e == out_e:
                            e.set(cur.next(e), cur)
                            logging.debug('[{}]: setting {}->{}'.format(cur.id, e.src.id, e.dst.id))
                    all_marked = True
                    for e in cur.edges:
                        logging.debug('[{}]: {} --> {}'.format(cur.id, cur.id, cur.next(e).id))
                        all_marked = all_marked & self.DFS(cur.next(e))
                        if not all_marked:
                            break
                    if all_marked:
                        logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                        assert self.stack.pop() == cur.id
                        cur.unmark()
                        return True
                    else:
                        for e in flexibles:
                            if not cur.next(e).marked:
                                logging.debug('[{}]: resetting {}->{}'.format(
                                    cur.id, e.src.id, e.dst.id))
                                e.reset()
                logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                assert self.stack.pop() == cur.id
                cur.unmark()
                return False
            else:
                logging.debug('[{}]: pop {}'.format(cur.id, cur.id))
                assert self.stack.pop() == cur.id
                cur.unmark()
                return False

    def convert_to_functional_graph_using_z3(self):
        self.graph.draw('Unmarked')
        input_map = {}
        for n in self.graph.node_set:
            if n.getType() == NodeType.INPUT:
                input_map[n.val] = True
            elif n.getType() == NodeType.VARIABLE:
                input_map[n.val] = False
            else:
                continue
        input_eq_map = dict([(n.id, n.val.names) for n in self.graph.getNextEqNode()])
        solution, time = graph_transform_z3(input_map, input_eq_map)
        logging.debug('z3 input map: {}'.format(input_map))
        logging.debug('z3 eq map: {}'.format(input_eq_map))
        if not solution:
            # This is either underdetermined or inconsistent.
            return False, time
        else:
            logging.debug('z3 solution: {}'.format(solution))
            # Update edges of equation nodes.
            for eq_id in list(solution.keys()):
                if not solution[eq_id]:
                    # Then this equation is a constraint.
                    for n in self.graph.node_set:
                        if n.id == eq_id:
                            n.setType(NodeType.CONSTRAINT)
                else:
                    for n in self.graph.node_set:
                        # Get the equation node.
                        if n.id == eq_id:
                            for e in n.edges:
                                # Find the outgoing variable node.
                                if n.next(e).val == solution[eq_id]:
                                    # Set direction fix to False so that we can use
                                    # the same graph for other tests.
                                    e.set(n, n.next(e), False)
                                else:
                                    e.set(n.next(e), n, False)
            # Update edges of constraint nodes.
            for n in self.graph.getNextConNode():
                for e in n.edges:
                    e.set(n.next(e), n, False)
            self.graph.check()
            self.graph.draw('z3_marked')
            return True, time

    def convert_to_functional_graph(self):
        # Unmark all nodes first because mark is also used for name cloning.
        for n in self.graph.node_set:
            if n.marked:
                n.unmark()

        # Marks input nodes.
        for n in self.graph.node_set:
            if n.getType() == NodeType.INPUT:
                for e in n.edges:
                    assert not e.fixed
                    e.set(n, n.next(e), True)
                n.mark()

        # Marks all constraint nodes.
        for n in self.graph.node_set:
            if n.getType() == NodeType.CONSTRAINT:
                for e in n.edges:
                    if e.fixed:
                        assert e.dst is n, 'conflicting edge {}->{}'.format(e.src.id, e.dst.id)
                    else:
                        e.set(n.next(e), n, True)
                n.mark()

        # Marks all dangling variable nodes.
        for n in self.graph.node_set:
            if n.getType() == NodeType.VARIABLE and len(n.edges) == 1:
                e = n.edges[0]
                e.set(n.next(e), n, True)
                n.mark()

        self.graph.draw('Unmarked')
        # Tries to convert all other edges.
        for cur in self.graph.getUnmarkedNode():
            logging.debug('Start convert at {}'.format(cur.id))
            if not self.DFS(cur):
                logging.log(logging.ERROR, 'Cannot evaluate, unsolvable dependency graph from node {}'.format(cur.id))
                return False

        # Convert equation nodes w/o out edges to constraints.
        for n in self.graph.node_set:
            if n.getType() == NodeType.EQUATION:
                outdegree = 0
                for e in n.edges:
                    assert e.isDirected()
                    if e.src is n:
                        outdegree += 1
                        break
                if outdegree == 0:
                    n.setType(NodeType.CONSTRAINT)

        self.graph.check()
        self.graph.draw('marked')
        return True

    def generate_functions(self):
        def can_split(symbol):
            if '.' in symbol:
                return False
            return _token_splittable(symbol)

        def do_filter(node):
            def strip_index(toks):
                indexing = 0
                func_str = ''
                for i, t in enumerate(toks):
                    if t == '[':
                        indexing = indexing + 1
                    if not indexing:
                        func_str = func_str + t
                    if t == ']':
                        indexing = indexing - 1
                return func_str

            def actual_func(cond_func, func, **kwargs):
                cond_filter = cond_func(**kwargs)
                filtered_args = {}
                for k, v in kwargs.items():
                    filtered_args[k] = [val for val, cond in zip(v, cond_filter) if cond]
                return func(**filtered_args)

            assert node.val.scripted
            start = node.val.toks.index(Names.listCond)
            end = node.val.toks.index(')', start)
            cond_toks = node.val.toks[start + 1:end]
            cond_str = strip_index(cond_toks)
            func_toks = node.val.toks[:start] + node.val.toks[end:]
            func_str = strip_index(func_toks)

            syms = {}
            for name in node.val.names:
                syms.update(SympyHelper.initSyms(name))
            cond_exprs = SympyHelper.initExprs([cond_str], syms)
            func_exprs = SympyHelper.initExprs([func_str], syms)
            cond_sol = simplify(cond_exprs)[0]
            func_sol = list(solve(func_exprs, exclude=list(SympyHelper.initSyms(node.ordered_given).values()),
                                  check=False, manual=True, rational=False)[0].values())[0]

            cond_func = lambdify(tuple(node.ordered_given), cond_sol,
                                 modules=[{'umin': umin, 'ufloor': ufloor}, mcerp.umath, 'numpy', 'sympy'])
            func = lambdify(tuple(node.ordered_given), func_sol,
                            modules=[{'umin': umin, 'ufloor': ufloor}, mcerp.umath, 'numpy', 'sympy'])
            node.func = functools.partial(actual_func, cond_func=cond_func, func=func)
            node.func_str = lambdastr(tuple(node.ordered_given), func_sol)

        def umin(a, b):
            if not isinstance(a, mcerp.UncertainFunction) and not isinstance(b, mcerp.UncertainFunction):
                return min(a, b)
            else:
                x = Dummy()
                if not isinstance(a, mcerp.UncertainFunction):
                    tmp_a = np.asarray([a] * len(x._mcpts))
                    tmp_b = b._mcpts
                elif not isinstance(b, mcerp.UncertainFunction):
                    tmp_a = a._mcpts
                    tmp_b = np.asarray([b] * len(x._mcpts))
                else:
                    raise ValueError('Should not be here, flip the table and run!')
                for i in range(len(x._mcpts)):
                    x._mcpts[i] = min(tmp_a[i], tmp_b[i])
                return x

        def ufloor(a):
            if not isinstance(a, mcerp.UncertainFunction):
                return np.floor(a)
            else:
                x = np.asarray([np.floor(i) for i in a._mcpts])
                a._mcpts = x
                return a

        def do_generation(cur, do_solve=False):
            assert isinstance(cur.val, Relation)
            syms = {}
            for name in cur.val.names:
                syms.update(SympyHelper.initSyms(name,
                                                 self.v2v[getBaseName(name)].vector and cur.val.deferred))
            exprs = SympyHelper.initExprs([cur.val.str], syms)
            # Sympy bug workaround:
            # Must set rational to False, otherwise piecewise function cannot be solved correctly.
            if do_solve:
                solutions = solve(exprs, exclude=list(SympyHelper.initSyms(cur.ordered_given).values()),
                                  check=False, manual=True, rational=False)
                solution = list(solutions[0].values())[0]
            else:
                solutions = simplify(exprs)
                solution = solutions[0]
            if len(solutions) != 1:
                # We cannot handle multiplte solutions yet.
                raise NotImplementedError
            cur.func = lambdify(tuple(cur.ordered_given), solution,
                                modules=[{'umin': umin, 'ufloor': ufloor}, mcerp.umath, 'numpy', 'sympy'])
            cur.func_str = lambdastr(tuple(cur.ordered_given), solution)

        # TODO: cycle elimination.

        # Turns equation into lambda function.
        for cur in self.graph.getNextEqNode():
            cur.ordered_given = []
            for e in cur.edges:
                assert e.isDirected()
                if e.dst is cur:
                    assert e.src.getType() == NodeType.VARIABLE or e.src.getType() == NodeType.INPUT
                    cur.ordered_given.append(e.src.val)
                else:
                    assert e.src is cur
                    assert e.dst.getType() == NodeType.VARIABLE
                    cur.setOutName(e.dst.val)
            if cur.val.scripted:
                do_filter(cur)
            else:
                # This is generating value, do solving.
                do_generation(cur, do_solve=True)

        # Turns constraint into lambda function.
        for cur in self.graph.getNextConNode():
            cur.ordered_given = []
            for e in cur.edges:
                assert e.isDirected()
                assert e.dst is cur
                assert e.src.getType() == NodeType.VARIABLE or e.src.getType() == NodeType.INPUT
                cur.ordered_given.append(e.src.val)
                cur.setOutName()
            if cur.val.scripted:
                # This is inequality or over-determined, do not solve.
                do_filter(cur)
            else:
                do_generation(cur)

    def evaluate_graph(self, node, k, val):
        """ Evaluates node and propogates its value.
        """
        if val is None:
            return
        if (isinstance(val, tuple) or isinstance(val, list)) and len(val) == 1:
            val = val[0]
        if node.getType() == NodeType.INPUT or node.getType() == NodeType.VARIABLE:
            assert k == node.val, 'Unknown variable {} encountered durig evaluation'.format(k)
            node.out_val = val
            for e in node.edges:
                assert e.isDirected()
                if e.src is node:
                    self.evaluate_graph(e.dst, node.out_name, node.out_val)
        elif node.getType() == NodeType.EQUATION:
            assert k in node.ordered_given, \
                'Unknown keyword {} propagated to node {}'.format(k, node.getPrintable())
            node.proped[k] = val
            # When we have all inputs ready.
            if set(node.ordered_given) == set(node.proped.keys()):
                logging.debug('evaluate: {}'.format(node.func_str))
                logging.debug('with: {}'.format(node.proped))
                for k, v in node.proped.items():
                    logging.debug('\t{}: {}'.format(k, type(v)))
                node.out_val = node.func(**node.proped)
                logging.debug('gen: {}={}'.format(node.out_name, node.out_val))
                self.evaluate_graph(self.v2n[node.out_name], node.out_name, node.out_val)
        else:
            assert node.getType() == NodeType.CONSTRAINT, \
                'Unknown node type {} when evaluatig node {}'.format(node.getPrintable())

    def build_dependency_graph(self):
        self.v2n = {}  # variable name -> graph node
        for v in self.var_set:
            self.v2n[v] = GraphNode(NodeType.VARIABLE, v)
            if hasExtName(v):
                self.v2n[v].addExtName(getExtName(v))
            # Variables produce their own value at evaluation.
            self.v2n[v].setOutName(v)
            self.v2n[v].vector = self.v2v[v].vector
            self.graph.addNode(self.v2n[v])

        # Add input nodes.
        for inputs in list(self.given.keys()):
            # After gen_input, each input is in the form of a tuple.
            for v in inputs:
                exist = False
                for n in self.graph.node_set:
                    if n.getType() == NodeType.VARIABLE and n.val == v:
                        exist = True
                        n.setType(NodeType.INPUT)
                if not exist:
                    self.v2n[v] = GraphNode(NodeType.INPUT, v)
                    if hasExtName(v):
                        self.v2n[v].addExtName(getExtName(v))
                    self.v2n[v].setOutName(v)
                    self.v2n[v].vector = self.v2v[getBaseName(v)].vector
                    self.graph.addNode(self.v2n[v])

        for e in self.eq_set:
            eq = GraphNode(NodeType.EQUATION, e)
            for v in e.names:
                assert v in self.v2n
                edge = GraphEdge(eq, self.v2n[v])
                eq.addEdge(edge)
                self.v2n[v].addEdge(edge)
                self.graph.addEdge(edge)
            self.graph.addNode(eq)

        for c in self.con_set:
            con = GraphNode(NodeType.CONSTRAINT, c)
            for v in c.names:
                assert v in self.v2n
                edge = GraphEdge(con, self.v2n[v])
                con.addEdge(edge)
                self.v2n[v].addEdge(edge)
                self.graph.addEdge(edge)
            self.graph.addNode(con)
        self.graph.draw('Before_clone')
        self.nameClone()

    def cloneNode(self, cur, exts):
        """ Create cloned nodes of cur with exts. Add them to graph.
        """
        cloned_set = set()
        # Create clone nodes.
        for ext in exts:
            cloned = cur.clone(ext, exts)
            self.graph.addNode(cloned)
            self.graph.addEdges(cloned.edges)
            assert cloned not in cloned_set
            cloned_set.add(cloned)
        assert cloned_set
        return cloned_set

    def splitNode(self, cur, ext_set):
        logging.debug('Split [{}] {}'.format(cur.id, cur.getPrintable()))
        if not cur in self.graph.node_set:
            return
        neighbours = [cur.next(e) for e in cur.edges
                      if not (set(cur.next(e).exts) & ext_set) and not cur.next(e).marked]
        if cur.getType() == NodeType.VARIABLE:
            cloned_set = self.cloneNode(cur, ext_set)
            self.graph.removeNode(cur)
            logging.debug('[{}] {} split into {}'.format(
                cur.id, cur.val, [(n.id, n.val) for n in cloned_set]))
            assert cur.val in self.var_set
            self.var_set.remove(cur.val)
            del self.v2n[cur.val]
            for n in cloned_set:
                self.var_set.add(n.val)
                self.v2n[n.val] = n
                self.v2n[n.val].setOutName(n.val)
            for n in neighbours:
                self.splitNode(n, ext_set)
        elif cur.getType() == NodeType.EQUATION or cur.getType() == NodeType.CONSTRAINT:
            cur.mark()
            for n in neighbours:
                self.splitNode(n, ext_set)
            cloned_set = self.cloneNode(cur, ext_set)
            self.graph.removeNode(cur)
            logging.debug('[{}] {} split into'.format(cur.id, cur.val.str))
            for n in cloned_set:
                logging.debug('\t[{}] {}'.format(n.id, n.val.str))
        else:
            # Propagation ends at inputs.
            assert cur.getType() == NodeType.INPUT
        return

    def nameClone(self):
        """ Name cloning: split the graph.
        1. Search all base names, if there are extended nodes:
            1.1. Split all variable nodes.
            1.2. Split all relation nodes.
        2. Merge variables with same names.
        3. Checks exts along the path and delete dummy edges.
        """
        # All base names with extended versions.
        basis = set([getBaseName(v) for v in self.var_set if hasExtName(v)])
        logging.debug('base names: {}'.format(basis))
        for base_name in basis:
            if base_name in self.var_set:
                # Split node.
                assert base_name in self.v2n
                cur = self.v2n[base_name]
                ext_set = self.findAllExtNames(base_name)
                for ext in ext_set:
                    full_name = concatExtName(base_name, ext)
                    assert full_name in self.v2n
                logging.debug('Start spliting from [{}] {}'.format(cur.id, cur.val))
                self.splitNode(cur, ext_set)
            # Otherwise, the extensios only exist in inputs, or the node has been split
            # via propagation.
        # Merge identical variable nodes.
        remove_set = set()
        for name in list(self.v2n.keys()):
            cur = self.v2n[name]
            for n in self.graph.node_set:
                if not n is cur and n.val == cur.val:
                    logging.debug('Merging {}'.format(n.val))
                    # The following is not true in pra model when the extensions
                    # appear in the same connected component as the base. They
                    # can be linked via some path (other variables in between).
                    # assert not self.graph.isConnected(n, cur)
                    for e in n.edges:
                        nn = n.next(e)
                        if e.isDirected():
                            if e.src is n:
                                new_edge = GraphEdge(cur, nn, cur, nn)
                            else:
                                assert e.dst is n
                                new_edge = GraphEdge(cur, nn, nn, cur)
                        else:
                            new_edge = GraphEdge(cur, nn)
                        cur.addEdge(new_edge)
                        nn.addEdge(new_edge)
                        self.graph.addEdge(new_edge)
                    if n.getType() == NodeType.INPUT:
                        cur.setType(NodeType.INPUT)
                    remove_set.add(n)
        # Remove dup nodes.
        for n in remove_set:
            self.graph.removeNode(n)

    def findAllExtNames(self, base_name):
        assert not hasExtName(base_name)
        exts = set()
        for var in self.var_set:
            if hasExtName(var) and base_name == getBaseName(var):
                exts.add(getExtName(var))
        assert exts
        logging.debug('All ext names for {}: {}'.format(base_name, exts))
        return exts

    def link(self):
        type_nodes = [n for n in self.nodes if isinstance(n, TypeNode)]
        for t in type_nodes:
            self.n2t[t.name] = t
        rule_nodes = [n for n in self.nodes if isinstance(n, RuleNode)]
        for r in rule_nodes:
            self.n2r[r.name] = r

        type_names = [n.name for n in type_nodes]
        for r in rule_nodes:
            # Checks type definition.
            for v in r.vars:
                assert v.type_name in type_names, 'Undefied type {} i rule {}'.format(
                    v.type_name, r.name)
                self.v2v[v.name] = v

            # Checks var definition.
            var_names = [v.name for v in r.vars]
            var_alias = [v.short_name for v in r.vars]
            for v in r.use:
                assert v in var_names or v in var_alias, 'Undefined var {} in rule {}'.format(
                    v, r.name)

        stmt_nodes = [n for n in self.nodes if n not in type_nodes and n not in rule_nodes]
        assumed_rules = []
        for s in stmt_nodes:
            if isinstance(s, AssumeNode):
                for r in s.assumed_rules:
                    assert r in self.n2r, 'Undefined rule {}'.format(r)
                assumed_rules += [self.n2r[r] for r in s.assumed_rules]
                for r in assumed_rules:
                    self.var_set.update([v.name for v in r.vars])
                    for v in r.vars:
                        if v.name in self.v2t:
                            assert v.type_name == self.v2t[v.name].name, \
                                'Inconsistent type: {} and {} for {}'.format(
                                    v.type_name, self.v2t[v.name].name, v.name)
                        else:
                            self.v2t[v.name] = self.n2t[v.type_name]
                    self.eq_set.update(r.equations)
                    self.con_set.update(r.constraints)
                    # Update assumptions from rule definition.
                    for k, v in r.given.items():
                        if k in self.given:
                            assert self.given[k] == v, \
                                'Redefinition of {} found'.format(k)
                        else:
                            self.assumptions[k] = v
            elif isinstance(s, LetNode):
                if not s.rset or s.rset < Names.builtin:
                    self.given[s.lvar] = s.rstr
                else:
                    self.dependent[s.lvar] = (s.rset - Names.builtin, s.rtoks)
                self.v2l[s.lvar] = s
            elif isinstance(s, SolveNode):
                if self.dependent:
                    for undecided, (dependents, rtoks) in self.dependent.items():
                        # Check that dependets are all given.
                        for var in dependents:
                            assert var in self.given, \
                                'Undecidable given value for {} due to {}'.format(
                                    undecided, var)
                        # Reduction.
                        new_toks = []
                        for t in rtoks:
                            if t in dependents:
                                new_toks.append(self.given[t])
                            else:
                                new_toks.append(t)
                        self.given[undecided] = ''.join(new_toks)
                given_keys = set()
                for k in list(self.given.keys()) + list(self.assumptions.keys()):
                    if isinstance(k, tuple):
                        given_keys.update(set([getBaseName(n) for n in k]))
                    else:
                        given_keys.add(getBaseName(k))
                for t in s.targets:
                    assert t in self.var_set, \
                        'Unknown variable to explore: {}'.format(t)
                    self.targets.append(t)
                assert self.targets, 'Empty solve target set.'
                # Check solvable.
                unique_variables = set([getBaseName(k) for k in self.var_set])
                n_free_variables = (len(unique_variables) -
                                    len(given_keys & unique_variables))
                n_equations = len(self.eq_set)
                if n_free_variables > n_equations:
                    logging.log(logging.ERROR, 'Underdetermined system,' \
                                               ' need {} more to be determined:\n{}'.format(
                        n_free_variables - n_equations,
                        '\t\n'.join(unique_variables -
                                    (given_keys & unique_variables))))
                if n_free_variables < n_equations:
                    logging.log(logging.ERROR, 'Overdetermined system, free {}, equation {}'.format(
                        n_free_variables, n_equations))
            elif isinstance(s, PlotNode):
                self.plot_nodes.append(s)
            else:
                raise ValueError('Unknown stmt: {}'.format(s.toks))

    def read_list_from_file(self, path):
        content = None
        with open(path, 'r') as ifile:
            content = ifile.readlines()
            ifile.close()
        return [int(i) for i in content]

    def gen_inputs(self):
        tup_given = {}
        for k, v in self.given.items():
            if self.v2l[k].path:
                val = self.read_list_from_file(self.v2l[k].path)
            else:
                val = eval(v)
            # Has to change range into list, otherwise fail for sympy. For example, range(1,50)**2 won't work
            if isinstance(val, range):
                val = list(val)

            if self.v2l[k].lvector:
                assert self.type_check(k, val), \
                    'Constraint for type {} not satisfied with value {}.'.format(
                        self.v2t[k].name, val)
                if not isinstance(k, tuple):
                    k = (k,)
                    val = (val,)
            elif isinstance(val, list):
                tval = []
                for one_val in val:
                    assert self.type_check(k, one_val), \
                        'Constraint for type {} not satisfied with value {}.'.format(
                            self.v2t[k].name, one_val)
                    tval.append((one_val,))
                if not isinstance(k, tuple):
                    k = (k,)
                    val = tval
            else:
                if k in self.v2t:
                    assert self.type_check(k, val), \
                        'Constraint for type {} not satisfied with value {}.'.format(
                            self.v2t[k].name, val)
                if not isinstance(k, tuple):
                    k = (k,)
                    val = [(val,)]
            tup_given[k] = val
        self.given = tup_given
        # Add assumptions to given dict.
        for k, v in self.assumptions.items():
            self.given[(k,)] = [(eval(v),)]

    def test_gc_overhead(self):
        # Test time overhead for graph conversion.
        dfs_time = 0.
        z3_time = 0.
        mm_time = 0.
        N = 10
        N_edges = 0
        S_density = 0
        for i in range(N):
            rg = RandomGraph(7, 7, sparse=True, drawable=self.drawable)
            self.graph = rg.getGraph()
            N_edges += len(self.graph.edge_set)
            if self.use_z3:
                solvable, time = self.convert_to_functional_graph_using_z3()
                z3_time += time
            for e in self.graph.edge_set:
                e.reset()
            start = timer()
            dfs_solvable = self.convert_to_functional_graph()
            end = timer()
            if self.use_z3:
                assert solvable == dfs_solvable
            dfs_time += end - start
            self.graph = rg.getNxGraph()
            S_density += nx.density(self.graph)
            start = timer()
            biGraph.maximum_matching(self.graph)
            end = timer()
            mm_time += end - start
        logging.log(logging.DEBUG, 'Avg # of edges: {}'.format(N_edges * 1. / N))
        logging.log(logging.DEBUG, 'Avg z3 time used: {}'.format(z3_time / N))
        logging.log(logging.DEBUG, 'Avg dfs time used: {}'.format(dfs_time / N))
        logging.log(logging.DEBUG, 'Avg mm time used: {}'.format(mm_time / N))
        return True

    def optimizeSMT(self, smt, knobs, k2s=None, minimize=True):
        """ Turns SMT solving to optimization by iteratively adding
            assumptions about 'knobs' with a step defined in dict k2s.

        Args:
            smt: an SMTInstance.
            knobs: list of variable names that we can tune.
            k2s: var_name -> step_value.
            minimize: direction of optimization.
        """

        assert smt, 'No smt instance to optimize.'
        assert knobs, 'Optimizing objective is emptyset.'
        smt.dump()
        solution = smt.solve()
        opt_solution = solution
        while solution:
            logging.debug('SMT OPT TRACE: {}'.format(solution))
            opt_solution = solution
            asmpts = []
            for k in knobs:
                op = '<' if minimize else '>'
                new_con = k + op + '(' + str(solution[k]) + '+' + str(k2s[k]) + ')'
                asmpts.append(new_con)
            logging.debug('SMT assumes\n\t[{}]'.format(',\n\t'.join(asmpts)))
            smt.makeAssumptions(asmpts)
            smt.dump()
            solution = smt.solve()
            smt.clearAssumptions()
        logging.log(logging.DEBUG, '{}: {}'.format('Opt solution', '[' + ',\n'.join(
            [tar + ': ' + str(opt_solution[tar]) for tar in self.targets]) + ']'))

    def constructSMTInstance(self):
        var_map = {}
        rel_list = []
        # Variable types.
        for var in list(self.v2t.keys()):
            var_map[var] = self.v2t[var].data_type
        # Relations.
        for n in self.graph.getNextRelationNode():
            rel_list.append(n.val.str)
        # Relations from let stmt.
        let_eqs = []
        mutable_eqs = []
        results = defaultdict(list)
        iter_vars, iter_vals = [], []
        flat_iter_vars = []
        for k, v in self.given.items():
            if isinstance(v, list):
                iter_vars.append(k)
                iter_vals.append(v)
                for var in k:
                    flat_iter_vars.append(var)
        if flat_iter_vars:
            logging.log(logging.DEBUG, "Result {}".format(tuple(flat_iter_vars)))

        # Handle flat variables.
        for key, val in self.given.items():
            for k, v in zip(key, val):
                # Propagate non-iterables first.
                if not k in flat_iter_vars and k in self.v2n:
                    let_eqs.append(k + '=' + str(v))

        rel_list.extend(let_eqs)
        return SMTInstance(var_map, rel_list)

    def solveSMT(self):
        var_map = {}
        rel_list = []
        # Variable types.
        for var in list(self.v2t.keys()):
            var_map[var] = self.v2t[var].data_type
        # Relations.
        for n in self.graph.getNextRelationNode():
            rel_list.append(n.val.str)
        # Relations from let stmt.
        let_eqs = []
        mutable_eqs = []
        results = defaultdict(list)
        iter_vars, iter_vals = [], []
        flat_iter_vars = []
        for k, v in self.given.items():
            if isinstance(v, list):
                iter_vars.append(k)
                iter_vals.append(v)
                for var in k:
                    flat_iter_vars.append(var)
        if flat_iter_vars:
            logging.log(logging.DEBUG, "Result {}".format(tuple(flat_iter_vars)))

        # Handle flat variables.
        for key, val in self.given.items():
            for k, v in zip(key, val):
                # Propagate non-iterables first.
                if not k in flat_iter_vars and k in self.v2n:
                    let_eqs.append(k + '=' + str(v))
        rel_list.extend(let_eqs)

        # Handle iterations.
        proped_vals = dict([(k, None) for k in flat_iter_vars])
        for t in itertools.product(*tuple(iter_vals)):
            mutable_eqs = []
            for key, val in zip(iter_vars, t):
                for k, v in zip(key, val):
                    mutable_eqs.append(k + '=' + str(v))
                    proped_vals[k] = v
            tag = tuple([proped_vals[var] for var in flat_iter_vars])
            smt = SMTInstance(var_map, rel_list + mutable_eqs)
            solution = smt.solve()
            self.result = {}
            for tar in self.targets:
                logging.log(logging.DEBUG, 'Result: {} -> {} = {}'.format(tag, tar, solution[tar]))
                self.result[tar] = solution[tar]

    def solveDetermined(self):
        results = defaultdict(list)
        iter_vars = []
        flat_iter_vars = []
        iter_vals = []
        for k, v in self.given.items():
            if isinstance(v, list):
                iter_vars.append(k)
                iter_vals.append(v)
                for var in k:
                    flat_iter_vars.append(var)
        if flat_iter_vars:
            logging.debug("Result {}".format(tuple(flat_iter_vars)))
        already_evaluated = set()
        proped_vals = dict([(k, None) for k in flat_iter_vars])
        last_proped_vals = None
        start = timer()
        # Changed the mechanism so that every variable is iterable, maybe not need the code below any more
        # for key, val in self.given.items():
        #     for k, v in zip(key, val):
        #         # Propagate non-iterables first.
        #         if not k in flat_iter_vars and k in self.v2n:
        #             self.evaluate_graph(self.v2n[k], k, v)
        # # If we are done with inputs.
        # if not iter_vars:
        #     if self.graph.eval_constraints():
        #         for tar in self.targets:
        #             logging.debug('Result {}: {}'.format(tar, self.v2n[tar].out_val))
        # Handle iterations.
        for t in itertools.product(*tuple(iter_vals)):
            for key, val in zip(iter_vars, t):
                # if not isinstance(val,list) or not isinstance(val,tuple):
                #     val=[val]
                for k, v in zip(key, val):
                    proped_vals[k] = v
                    tag = tuple([proped_vals[var] for var in flat_iter_vars])
                if tag not in already_evaluated:
                    already_evaluated.add(tag)
                    difference = dict([(tmp_k, proped_vals[tmp_k])
                                       for tmp_k in proped_vals if
                                       (last_proped_vals[tmp_k] is None or
                                        proped_vals[tmp_k] != last_proped_vals[tmp_k]) and
                                       tmp_k in self.v2n]) if last_proped_vals else proped_vals
                    # Explicitly make a copy.
                    last_proped_vals = dict(proped_vals)
                    for k in difference:
                        # for k in proped_vals: # w/ redundant computation
                        self.evaluate_graph(self.v2n[k], k, proped_vals[k])
                    if self.graph.eval_constraints():
                        for tar in self.targets:
                            if self.type_check(tar, self.v2n[tar].out_val):
                                # results[tar].append((tag, self.v2n[tar].out_val))
                                results[tag].append(self.v2n[tar].out_val)

                                logging.info('Result {} -> {} = {}'.format(tag, tar, self.v2n[tar].out_val))
        end = timer()

        self.result = results
        self.variables = iter_vars
        self.values = iter_vals
        self.flat_variables = flat_iter_vars
        # logging.log(logging.INFO, 'Results saved to {}'.format(file_name))
        # logging.log(logging.INFO, 'Time used: {}'.format(end - start))

    def __plot(self, node: PlotNode):
        if node.dependent not in self.targets:
            logging.error("Var {} not in explored targets, cannot be plotted".format(node.dependent))
            return
        all_variables = list(self.flat_variables) + list(self.targets)
        all_values = [tuple(k) + tuple(self.result[k]) for k in self.result]
        plot_data = defaultdict(list)
        correlated_with_free_variables = []
        # i.e. if 'assume (a,b)=[(?,?),...]', a is the free variable, the b is considered correlated
        # correlated variables should be treated differently, as they change with the free variable
        for vars in self.given.keys():
            for free_var in node.free:
                if free_var in vars:
                    correlated_with_free_variables += list(vars)
        # Compute values for unrelated variables
        for value in all_values:
            tag = []
            for k, v in zip(all_variables, value):
                if k not in correlated_with_free_variables:
                    if k in node.given_var_dict:
                        if v not in node.given_var_dict[k]:
                            break
                    tag.append(v)
            else:
                plot_data[tuple(tag)].append(
                    [value[all_variables.index(i)] for i in node.free] + [value[all_variables.index(node.dependent)]]
                )
        handles = []
        legends = []
        # Iterate through each possible combination of values of unrelated variables and plot separately each
        if len(node.free) == 1:
            ax = plt.subplot(111)
        else:
            fig = plt.figure()
            ax = Axes3D(fig)
        for tag in plot_data:
            xs = [i[:-1] for i in plot_data[tag]]
            y = [i[-1] for i in plot_data[tag]]
            if len(xs) > 0:
                if len(xs[0]) == 1:
                    handles.append(getattr(ax, node.plot_type)(xs, y))
                    legends.append(tag)
                else:
                    handles.append(getattr(ax, node.plot_type)([i[0] for i in xs], [i[1] for i in xs], y))
        if len(handles) > 0:
            if len(node.free) == 1:
                ax.set_xlabel(node.free[0])
                ax.set_ylabel(node.dependent)
            else:
                ax.set_xlabel(node.free[0])
                ax.set_ylabel(node.free[1])
                ax.set_zlabel(node.dependent)
            image = io.BytesIO()
            plt.savefig(image, dpi='figure')
            plt.clf()
            return image

    def run(self):
        self.link()
        self.gen_inputs()
        self.build_dependency_graph()

        use_smt = False
        if self.use_z3:
            consistent_and_determined, _ = self.convert_to_functional_graph_using_z3()
            if not consistent_and_determined:
                use_smt = True
        else:
            consistent_and_determined = self.convert_to_functional_graph()

        if not consistent_and_determined and not use_smt:
            logging.log(logging.FATAL, 'System underdetermined or inconsistent, ''and not using z3 core, aborting '
                                       'evaluation...')
        elif consistent_and_determined and not use_smt:
            self.generate_functions()
            self.solveDetermined()
        elif not consistent_and_determined and use_smt:
            logging.log(logging.ERROR,
                        'System underdetermined or inconsistent, ''trying to solve as an SMT instance...')
            self.solveSMT()

        self.images = []
        if self.plot_nodes:
            for node in self.plot_nodes:
                self.images.append(self.__plot(node).getvalue())
        return {
            'raw': self.result,
            'img': self.images
        }

    def save(self):
        file_name = '-'.join(self.targets)
        file_name += '-ON-' + '-'.join(self.flat_variables) + '.out'
        with open(file_name, 'wb') as ofile:
            pickle.dump(self.result, ofile)
            ofile.close()
        for node, img in zip(self.plot_nodes, self.images):
            filename = '{}_against_{}'.format(node.dependent, node.free)
            with open(filename, 'wb') as f:
                f.write(img)
