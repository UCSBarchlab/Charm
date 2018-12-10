#!/usr/bin/env python
import warnings

from pyparsing import *

from .interpreter import *

ParserElement.setDefaultWhitespaceChars(' ')

args = None




class Program:
    def __init__(self, source, args=None, callback=lambda x: None):
        indent_stack = [1]
        all_names = '''
        typeDef
        ruleDef
        givenStmt
        assumeStmt
        solveStmt
        importStmt
        plotBlock
        '''.split()
        COND_EQ = re.compile(r',\s*(\w+)\s*=\s*(\w+)\s*\)')
        NAME_EXT = re.compile(r'(?![\d.+])([\w]+)\.([\w.]+)')
        # Convert '=' to Eq in piecewise function, otherwise sympy cannot parse correctly.
        source = COND_EQ.sub(lambda p: ', Eq(' + p.group(1) + ', ' + p.group(2) + '))', source)
        # Convert '.' expresion to custom name clone extension
        # because sympy cannot accept '.' in variable names.
        source = NAME_EXT.sub(lambda p: p.group(1) + Names.clone_ext + \
                                        Names.clone_ext.join(p.group(2).split('.')), source)
        self.imported = []
        self.args = args
        self.callback = callback
        COLON = Literal(':').suppress()
        COMMA = Literal(',')
        LPA = Literal('(')
        RPA = Literal(')')
        BS = Literal('/')
        ENDL = LineEnd().suppress()
        pyType = Literal('int') ^ Literal('float')
        equalityOp = Literal(Names.equality)
        inequalityOp = Literal('<') ^ Literal('>') ^ Literal('<=') ^ Literal('>=')
        operator = (Literal('+') ^ Literal('-') ^ Literal('*') ^
                    Literal('/') ^ Literal('**')).setName('op')
        typeExt = Literal('+') ^ Literal('-')
        term = Word(alphanums + '_' + '.' + '-')
        unit = Group(OneOrMore(Word(alphanums + '()*/^')))
        path = Word(alphanums + "\\/._-")
        component = Forward().setName('component')
        expression = Forward().setName('expression')
        constraint = Forward().setName('constraint')
        arg = Forward().setName('arg')
        arglist = Forward().setName('arglist')
        func = (term + arglist).setName('func')
        varName = (term + Optional(Literal('[') + Literal(']'))).setName('varName')
        equation = (expression + equalityOp + (expression ^ func)).setName('equation')
        constraint << (expression + inequalityOp + expression +
                       Optional((Literal('&&') | Literal('||')) + constraint))
        typeName = Group(term + Optional(typeExt))

        typeDecl = (Literal('typedef') + typeName(Names.typeName) + COLON +
                    pyType(Names.pyTypeName) + term(Names.shortName) + ENDL)
        typeBody = indentedBlock(constraint(Names.constraint) ^ equation(Names.equation),
                                 indent_stack)(Names.typeBody)
        # TODO maybe implement physical unit wildcard and infer in the future
        varDecl = (varName(Names.varName) + COLON + typeName(Names.typeName) +
                   Optional(Literal('as') + term(Names.shortName)) + Optional(
                    Keyword("in").suppress() + unit(Names.var_unit)))

        ruleDecl = (Literal('define') + term(Names.ruleName) + COLON + ENDL)
        ruleBody = indentedBlock(varDecl ^ constraint(Names.constraint) ^ equation(Names.equation),
                                 indent_stack)(Names.ruleBody)

        list_struct = Optional(term) + Literal('[') + arg + \
                      ZeroOrMore(COMMA + arg) + Literal(']') + \
                      Optional(Literal('|') +
                               (equation ^ constraint))
        tuple_struct = (LPA + arg + ZeroOrMore(COMMA + arg) + RPA).setName('tuple')
        struct = (list_struct ^ tuple_struct).setName('struct')

        component << (term ^ varName ^ func ^ struct)
        var = arg << (component ^ expression ^ equation ^ struct ^ constraint)
        arglist << (LPA + Optional(arg + ZeroOrMore(COMMA + arg)) + RPA)
        expression << ((Optional(BS) + component +
                        ZeroOrMore(operator + expression)) ^ (LPA + expression + RPA))
        if args.verbose:
            self.typeDef.setDebug()
            self.ruleDef.setDebug()
            self.blankStmt.setDebug()
            self.givenStmt.setDebug()
            self.assumeStmt.setDebug()
            self.solveStmt.setDebug()
            func.setDebug()
            equation.setDebug()
            constraint.setDebug()
            arglist.setDebug()
            arg.setDebug()
            expression.setDebug()
            component.setDebug()
            inequalityOp.setDebug()
            struct.setDebug()

        self.givenStmt = Group(Literal('given') +
                               OneOrMore(term +
                                         Optional(COMMA))(Names.assumedRule)).setResultsName(Names.assume, True)
        self.assumeStmt = (Suppress('assume') + equation +
                           Optional(ZeroOrMore(COMMA.suppress() + equation))).setResultsName(Names.let, True)
        self.solveStmt = Group(Literal('explore') +
                               OneOrMore(term +
                                         Optional(COMMA.suppress()))(Names.target)).setResultsName(Names.solve, True)
        self.typeDef = Group(typeDecl + typeBody).setResultsName(Names.typeDef, True)
        self.ruleDef = Group(ruleDecl + ruleBody).setResultsName(Names.ruleDef, True)
        self.analysisStmt = (self.givenStmt | self.assumeStmt | self.solveStmt) + ENDL
        self.blankStmt = Suppress((LineStart() + LineEnd()) ^ White()).setName('blankStmt')
        self.commentStmt = (Literal('#') + restOfLine + ENDL).setName('comment')
        plot_given_condition = Group(term + Literal("=").suppress() + list_struct)
        plot_free_variable_list = Group(term + Optional(COMMA.suppress() + term))(Names.plot_free_variable)
        plot_conditions = indentedBlock(OneOrMore(plot_given_condition + ENDL), indent_stack)(
            Names.plot_given_condition)
        self.plotBlock = Keyword("plot").suppress() + term(Names.plot_dependent_variable) \
                         + Keyword("against").suppress() + plot_free_variable_list \
                         + Keyword("as").suppress() + term(Names.plot_type) \
                         + Optional(Keyword("where").suppress() + Literal(":").suppress() + ENDL + plot_conditions)
        direct_import_statement = Group(
            Keyword("import") +
            path.setResultsName(Names.import_path)
        ).setResultsName(Names.import_result_name)
        from_import_statement = Group(
            Keyword("from")
            + path.setResultsName(Names.import_path)
            + Keyword("import")
            + Group(term + ZeroOrMore(
                Literal(',').suppress() + term
            )).setResultsName(Names.import_modules)
            + Optional(Keyword("as") +
                       Group(
                           term + ZeroOrMore(
                               Literal(',').suppress() + term
                           )
                       ).setResultsName(Names.import_alias))
        ).setResultsName(Names.import_result_name)
        self.importStmt = (direct_import_statement | from_import_statement) + ENDL
        stmt = self.typeDef | self.ruleDef | self.analysisStmt \
               | self.blankStmt | self.commentStmt | self.importStmt | self.plotBlock

        self.program = OneOrMore(stmt)
        self.program.ignore(self.commentStmt)

        self.ast_nodes = []
        self.rule_nodes = []
        self.type_nodes = []

        def do_solveStmt(s, l, t):
            self.ast_nodes.append(SolveNode(t.asDict()))
            return

        def do_assumeStmt(s, l, t):
            self.ast_nodes.append(LetNode(t.asDict()))
            return

        def do_givenStmt(s, l, t):
            self.ast_nodes.append(AssumeNode(t.asDict()))
            return

        def do_typeDef(s, l, t):
            self.ast_nodes.append(TypeNode(t.asDict()))
            self.type_nodes.append(TypeNode(t.asDict()))
            return

        def do_ruleDef(s, l, t):
            self.ast_nodes.append(RuleNode(t.asDict()))
            self.rule_nodes.append(RuleNode(t.asDict()))
            return

        def do_importStmt(s, l, t):
            res = t.asDict()[Names.import_result_name]
            if res[Names.import_path] in self.imported:
                warnings.warn(
                    """
                    Duplicate import
                    """,
                    SyntaxWarning
                )
                # TODO develop a uniform way of mention syntax errors and warnings
                return
            try:
                self.imported.append(res[Names.import_path])
                src = open(res[Names.import_path] + '.charm', 'r').read()
                _imported = Program(src, args)
                types = _imported.type_nodes
                rules = _imported.rule_nodes
                if Names.import_modules in res and not '*' in list(res[Names.import_modules]):
                    _rules = rules
                    rules = []
                    for name in list(res[Names.import_modules]):
                        for rule in _rules:
                            if rule.name == name:
                                rules.append(rule)
                                break
                if Names.import_alias in res:
                    for i in range(min(len(res[Names.import_modules]), len(res[Names.import_alias]))):
                        rules[i].name = res[Names.import_alias][i]
                self.rule_nodes += rules
                self.type_nodes += types
                self.ast_nodes += types + rules
            except ParseException as e:
                logging.fatal("Fatal error:\n{}\n{}\n{}".format(e.line, " " * (e.column - 1) + "^", e))
                raise
            except Exception as e:
                raise SyntaxError(
                    """
                    Fatal error: import file failed(error message:{})
                    At: {}
                    
                    """.format(e, s)
                )

        def do_plotBlock(s, l, t):
            self.ast_nodes.append(PlotNode(t))

        for name in all_names:
            ex = vars(self)[name]
            action = vars()['do_' + name]
            ex.setName(name)
            ex.setParseAction(action)
        try:
            self.program.parseString(source, parseAll=True)
        except ParseException as err:
            logging.fatal("Fatal error:\n{}\n{}\n{}".format(err.line, " " * (err.column - 1) + "^", err))
            raise

    def run(self, save=False):
        class _Nodes(object):
            def __init__(self, nodes):
                self.nodes = nodes  # All ast nodes.

            def dump(self):
                for n in self.nodes:
                    n.dump()

        program = _Nodes(self.ast_nodes)
        interp = Interpreter(program, self.args.z3core, self.args.draw, self.args.mcsamples, self.callback)
        # interp.test_gc_overhead()
        result = interp.run()
        if save:
            return interp.save()
        return result