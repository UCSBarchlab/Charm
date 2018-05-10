import re
from sympy.solvers import solve
from sympy.parsing.sympy_parser import parse_expr, auto_symbol, convert_equals_signs

# A valid variable name consists alphanums and dot (excluding leading nums and dots).
VAR_NAME = re.compile(r'(?![\d\.+])[\w\.]+')

def extract_variables(text):
    return [v for v in VAR_NAME.findall(text) if not v in Names.builtin]

class Names(object):
    """ Reserved strings used in parser.
    """
    assume = 'assume_stmt'
    assumedRule = 'assumed'
    clone_ext = '___'
    constraint = 'constraint'
    equality = '='
    equation = 'equation'
    from_file='File'
    let = 'let_stmt'
    listCond = '|'
    listConstruct = '['
    metricBody = 'metric_body'
    metricDef = 'metric_def'
    metricName = 'metric_name'
    norm_dist = 'Gauss'
    piecewise = 'Piecewise'
    product = 'Product'
    pyTypeName = 'py_type'
    ruleBody = 'rule_body'
    ruleDef = 'rule_def'
    ruleName = 'rule_name'
    shortName = 'short_name'
    solve = 'solve_stmt'
    summation = 'Sum'
    max_func = 'Max'
    min_func = 'Min'
    target = 'to_solve'
    typeBody = 'type_body'
    typeDef = 'type_def'
    typeName = 'type_name'
    varName = 'var_name'
    # Function keywords. 
    builtin = set(['Eq', 'exp', 'log', norm_dist, 'range', 'floor', 'ceiling', summation,
        product, 'floor', 'ufloor', 'ceiling', 'uceiling', min_func, 'umin', max_func, 'umax',
        from_file, piecewise, 'list'])

STOPWORDS = [',', '(', ')']

class IdObject(object):
    """ Object with monotonically incrementing ID.
    """

    iid = 0
    def __init__(self):
        self.id = IdObject.iid
        IdObject.iid += 1

class Node(IdObject):
    def __init__(self):
        super(Node, self).__init__()

    def visit(self, visitor_func):
        visitor_func(self.root)

    def parse(self):
        raise NotImplementedError

    def exportZ3(self):
        raise NotImplementedError

    def dump(self, indent='', printable=True):
        raise NotImplementedError

class TypeNode(Node):
    """ Type definition node.

    Fields:
        toks: original parsed tokens.
        name: string format type name.
        data_type: correspoding python data type in string format.
        short_name: alias used in constraints.
        constraints: list of string formatted constraints.
        use: variable name used in constraints, should be identical to short_name.
    """
    def __init__(self, toks):
        super(TypeNode, self).__init__()
        self.toks = toks
        self.parse()

    def dump(self, indent='', printable=True):
        dump_str = indent + '[{}] TypeNode:\n'.format(self.id)
        dump_str += indent + '\tname: {}\n'.format(self.name)
        dump_str += indent + '\tdtype: {}\n'.format(self.data_type)
        dump_str += indent + '\tconstraints:\n'
        for con in self.constraints:
            dump_str += indent + '\t\t{}\n'.format(con)
        dump_str += indent + '\tuse: {}\n'.format(self.use)
        if (printable):
            print(dump_str)
        return dump_str

    def parse(self):
        assert Names.typeDef in self.toks
        assert len(self.toks[Names.typeDef]) == 1
        k2v = self.toks[Names.typeDef][0]
        self.name = k2v[Names.typeName]
        if len(self.name) == 1:
            self.name = self.name[0]
        elif len(self.name) == 2:
            self.name = self.name[0] + self.name[1]
        else:
            raise ValueError('Illegal type name {}'.format(self.name))
        self.data_type = k2v[Names.pyTypeName]
        self.short_name = k2v[Names.shortName]
        self.constraints = []
        self.use = None
        constraints = k2v[Names.typeBody]
        for con in constraints:
            self.constraints.append(''.join(con[Names.constraint]))
            use_list = extract_variables(self.constraints[-1])
            assert len(use_list) == 1, \
                    'More than one free variable used in typeDef: {}.'.format(use_list)
            self.use = use_list[0]
            assert self.use == self.short_name, \
                    'Inconsistent name used in typeDecl ({}) and typeDef ({})'.format(
                            self.short_name, self.use)

class VarNode(Node):
    """ Variable declaration node.

    Fields:
        toks: original parsed tokens.
        name: full variable name, should be globally consistent.
        type_name: name of the type this variable belongs.
        short_name: alias used in rule define.
    """
    def __init__(self, toks):
        super(VarNode, self).__init__()
        self.toks = toks
        self.vector = False
        self.parse()

    def dump(self, indent='', printable=True):
        dump_str = indent + '[{}] VarNode:\n'.format(self.id)
        dump_str += indent + '\tname: {}\n'.format(self.name)
        dump_str += indent + '\ttype_name: {}\n'.format(self.type_name)
        dump_str += indent + '\tshort_name: {}\n'.format(self.short_name)
        if (printable):
            print(dump_str)
        return dump_str

    def parse(self):
        self.name = self.toks[Names.varName]
        if len(self.name) > 1:
            self.vector = True
        self.name = self.name[0]
        type_name = self.toks[Names.typeName]
        if len(type_name) == 1:
            self.type_name = type_name[0]
        elif len(type_name) == 2:
            self.type_name = type_name[0] + type_name[1]
        else:
            raise ValueError('Illegal type name {} for var {}'.format(type_name, self.name))
        if Names.shortName in self.toks:
            self.short_name = self.toks[Names.shortName]
        else:
            self.short_name = self.name

class RuleNode(Node):
    """ Rule definition node.

    Fields:
        toks: original parsed tokens.
        name: full name of the rule, should be globally consistent and unique.
        vars: list of VarNode for all variables used in this rule.
        given: dict of variable full name to value.
        equations: list of string formatted equations of this rule.
        constraints: list of string formatted constraints of this rule.
        use: list of variable full names used in equations and constraints.
    """
    def __init__(self, toks):
        super(RuleNode, self).__init__()
        self.toks = toks
        self.a2n = {} # Alias to real variable names.
        self.given = {}
        self.parse()

    def dump(self, indent='', printable=True):
        dump_str = indent + '[{}] RuleNode:\n'.format(self.id)
        dump_str += indent + '\tname: {}\n'.format(self.name)
        for v in self.vars:
            dump_str += v.dump('\t', False)
        dump_str += indent + '\tconstraints:\n'
        for con in self.constraints:
            dump_str += indent + '{}\n'.format(con.dump('\t\t', False))
        dump_str += indent + '\tequations:\n'
        for eq in self.equations:
            dump_str += indent + '{}\n'.format(eq.dump('\t\t', False))
        dump_str += indent + '\tuse set: {}\n'.format(self.use)
        if self.given:
            dump_str += indent + '\tgiven dict: {}\n'.format(self.given)

        if (printable):
            print(dump_str)
        return dump_str

    def parse(self):
        assert Names.ruleDef in self.toks
        assert len(self.toks[Names.ruleDef]) == 1
        k2v = self.toks[Names.ruleDef][0]
        self.name = k2v[Names.ruleName]
        self.vars = []
        self.constraints = []
        self.equations = []
        self.use = set()
        for expr in k2v[Names.ruleBody]:
            if Names.varName in expr:
                self.vars.append(VarNode(expr))
                var = self.vars[-1]
                assert not var.name in self.a2n, \
                        'duplicate variable {} in {}'.format(
                                var.name, self.name)
                self.a2n[var.name] = var.name
                assert not var.short_name in self.a2n or var.short_name == var.name, \
                        'duplicate variable {} in {}'.format(
                                var.short_name, self.name)
                self.a2n[var.short_name] = var.name

        for expr in k2v[Names.ruleBody]:
            if Names.shortName in expr:
                assert expr[Names.shortName] in self.a2n
            elif Names.constraint in expr:
                flatten = expr[Names.constraint]
                self.constraints.append(Relation(flatten, self.a2n))
                #self.use.update([self.a2n[a] for a in extract_variables(self.constraints[-1].str)])
                self.use.update(self.constraints[-1].names)
            elif Names.equation in expr:
                flatten = expr[Names.equation]
                eq_str = ''.join(flatten)
                short_names = set(extract_variables(eq_str))
                if len(short_names) == 1:
                    # Handle assumptions, e.g. x = 1
                    val = solve(parse_expr(eq_str,
                        transformations=(convert_equals_signs,auto_symbol,)))
                    if isinstance(val, list):
                        self.given[self.a2n[short_names.pop()]] = \
                                str(val[0]) if len(val) == 1 else str(val)
                    else:
                        if not val:
                            raise ValueError('Unsat equation: {} in rule {}'.format(
                                ''.join(flatten), self.name))
                        else:
                            print 'WARNING: infinitely underdetermined equation' \
                                    'ignored: {} in rule "{}"'.format(
                                    ''.join(flatten), self.name)
                else:
                    self.equations.append(Relation(flatten, self.a2n))
                    #self.use.update([self.a2n[a] for a in extract_variables(eq_str)])
                    self.use.update(self.equations[-1].names)
            else:
                assert Names.varName in expr, 'Unknown expr in rule def: {}'.format(expr)

class AssumeNode(Node):
    """ Assumed models.
    """

    def __init__(self, toks):
        super(AssumeNode, self).__init__()
        self.toks = toks
        self.parse()

    def dump(self, indent='', printable=True):
        dump_str = indent + '[{}] AssumeNode:\n'.format(self.id)
        for r in self.assumed_rules:

            dump_str += indent + '\t{}\n'.format(r)
        if (printable):
            print(dump_str)
        return dump_str
    
    def parse(self):
        assert Names.assume in self.toks
        rules = self.toks[Names.assume][0]
        assert Names.assumedRule in rules
        self.assumed_rules = [r for r in rules[Names.assumedRule] if r not in STOPWORDS]

class LetNode(Node):
    """ Let assignment node.
    """
    def __init__(self, toks):
        super(LetNode, self).__init__()
        self.toks = toks
        self.lvector = False
        self.path = None
        self.parse()

    def dump(self, indent='', printable=True):
        dump_str = indent + '[{}] LetNode:\n'.format(self.id)
        dump_str += indent + '\t{} ({}) <- {} ({})\n'.format(self.lstr, self.lvar, self.rstr, self.rset)
        if (printable):
            print dump_str
        return dump_str

    def parse(self):
        assert Names.let in self.toks
        assign_toks = self.toks[Names.let][0]
        assign_str = ''.join(assign_toks)
        self.lstr = assign_str.split(Names.equality)[0]
        if self.lstr.endswith('[]'):
            self.lvector = True
        self.lvar = extract_variables(self.lstr)
        if len(self.lvar) > 1:
            # Tuple case.
            self.lvar = tuple(self.lvar)
        else:
            # Single assignment case.
            assert len(self.lvar) == 1, 'Unknown assignment lvar: {}'.format(self.lvar)
            self.lvar = self.lvar[0]
        self.rtoks = assign_toks[assign_toks.index('=')+1:]
        self.rstr = assign_str.split(Names.equality)[1]
        self.rset = set(extract_variables(self.rstr))
        if Names.from_file in self.rtoks:
            self.path = ''.join(self.rtoks[self.rtoks.index('(')+1:-1])
            self.rset = set([Names.from_file])

class SolveNode(Node):
    """ Explore node.
    """
    def __init__(self, toks):
        super(SolveNode, self).__init__()
        self.toks = toks
        self.parse()

    def dump(self, indent='', printable=True):
        dump_str = indent + '[{}] SolveNode:\n'.format(self.id)
        for t in self.targets:
            dump_str += indent + '\t{}\n'.format(t)
        if (printable):
            print dump_str
        return dump_str

    def parse(self):
        assert Names.solve in self.toks
        solve_stmt = self.toks[Names.solve][0]
        assert Names.target in solve_stmt
        self.targets = solve_stmt[Names.target]

class Relation(Node):
    """ Ralation node, including both equations and inqualities.
    """
    def __init__(self, toks, a2n):
        super(Relation, self).__init__()
        self.deferred = False
        self.scripted = False
        self.orig = ''.join(toks)
        # Whether to check for index variables.
        strict_check = True
        if Names.summation in toks or \
                Names.product in toks:
            self.deferred = True
        if Names.listCond in toks:
            self.scripted = True
        strict_check = not self.deferred and not self.scripted
        self.alias = set(extract_variables(self.orig))
        if not strict_check:
            self.alias = self.alias & set(a2n.keys())
        for alias in self.alias:
            assert alias in a2n, \
                    '{} not defined in equation {}'.format(
                            alias, self.orig)
        self.names = set([a2n[a] for a in self.alias])
        for a in self.alias:
            self.names.add(a2n[a])
        self.toks = []
        for t in toks:
            if t in self.alias:
                self.toks.append(a2n[t])
            else:
                self.toks.append(t)
        self.str = ''.join(self.toks)

    def subs(self, ext_name):
        base_name = ext_name[:ext_name.find(Names.clone_ext)]
        assert base_name in self.names and not ext_name in self.names, ext_name
        self.names.remove(base_name)
        self.names.add(ext_name)
        for i in xrange(len(self.toks)):
            if self.toks[i] == base_name:
                self.toks[i] = ext_name
        self.str = ''.join(self.toks)

    def dump(self, indent='', printable=True):
        dump_str = indent + self.str + '\n'
        #dump_str += indent + '\talias set: {}\n'.format(self.alias)
        #dump_str += indent + '\tname set: {}\n'.format(self.names)
        if printable:
            print dump_str
        return dump_str

class Program(object):
    def __init__(self, nodes):
        self.nodes = nodes # All ast nodes.

    def dump(self):
        for n in self.nodes:
            n.dump()
