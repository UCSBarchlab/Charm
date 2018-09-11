#!/usr/bin/env python
import argparse
from utils.charm_options import *
from abstract_syntax_tree import *
from interpreter import *
from pyparsing import *

ParserElement.setDefaultWhitespaceChars(' ')

COND_EQ = re.compile(r',\s*(\w+)\s*=\s*(\w+)\s*\)')
NAME_EXT = re.compile(r'(?![\d\.+])([\w]+)\.([\w\.]+)')

args = None

indentStack = [1]

all_names = '''
program
typeDef
ruleDef
givenStmt
assumeStmt
solveStmt
'''.split()

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
        indentStack)(Names.typeBody)
typeDef = Group(typeDecl + typeBody).setResultsName(Names.typeDef, True)

varDecl = (varName(Names.varName) + COLON + typeName(Names.typeName) +
        Optional(Literal('as') + term(Names.shortName)))

ruleDecl = (Literal('define') + term(Names.ruleName) + COLON + ENDL)
ruleBody = indentedBlock(varDecl ^ constraint(Names.constraint) ^ equation(Names.equation),
        indentStack)(Names.ruleBody)
ruleDef = Group(ruleDecl + ruleBody).setResultsName(Names.ruleDef, True)

list_struct = Optional(term) + Literal('[') + arg + \
        ZeroOrMore(COMMA + arg) + Literal(']') + \
        Optional(Literal('|') +
                (equation ^ constraint))
tuple_struct = (LPA + arg + ZeroOrMore(COMMA + arg) + RPA).setName('tuple')
struct = (list_struct ^ tuple_struct).setName('struct')

givenStmt = Group(Literal('given') +
        OneOrMore(term +
            Optional(COMMA))(Names.assumedRule)).setResultsName(Names.assume, True)
assumeStmt = (Suppress('assume') + equation +
        Optional(ZeroOrMore(COMMA.suppress() + equation))).setResultsName(Names.let, True)
solveStmt = Group(Literal('explore') +
        OneOrMore(term +
            Optional(COMMA.suppress()))(Names.target)).setResultsName(Names.solve, True)
analysisStmt = (givenStmt | assumeStmt | solveStmt) + ENDL
blankStmt = Suppress((LineStart() + LineEnd()) ^ White()).setName('blankStmt')
commentStmt = (Literal('#') + restOfLine + ENDL).setName('comment')
stmt = typeDef | ruleDef | analysisStmt | blankStmt | commentStmt
program = OneOrMore(stmt)
component << (term ^ varName ^ func ^ struct)
arg << (component ^ expression ^ equation ^ struct ^ constraint)
arglist << (LPA + Optional(arg + ZeroOrMore(COMMA + arg)) + RPA)
expression << ((Optional(BS) + component +
    ZeroOrMore(operator + expression)) ^ (LPA + expression + RPA))

program.ignore(commentStmt)

ast_nodes = []

def do_solveStmt(s, l, t):
    ast_nodes.append(SolveNode(t.asDict()))
    return

def do_assumeStmt(s, l, t):
    ast_nodes.append(LetNode(t.asDict()))
    return

def do_givenStmt(s, l, t):
    ast_nodes.append(AssumeNode(t.asDict()))
    return

def do_typeDef(s, l, t):
    ast_nodes.append(TypeNode(t.asDict()))
    return

def do_ruleDef(s, l, t):
    ast_nodes.append(RuleNode(t.asDict()))
    return

def do_program(s, l, t):
    program = Program(ast_nodes)
    interp = Interpreter(program, args.z3core, args.draw, args.mcsamples)
    #interp.test_gc_overhead()
    interp.run()
    return

for name in all_names:
    ex = vars()[name]
    action = vars()['do_' + name]
    ex.setName(name)
    ex.setParseAction(action)

def main():
    global args
    parser = get_parser()
    addCommonOptions(parser)
    addCompilerOptions(parser)
    args = parse_args(parser)

    if args.verbose:
        typeDef.setDebug()
        ruleDef.setDebug()
        blankStmt.setDebug()
        givenStmt.setDebug()
        assumeStmt.setDebug()
        solveStmt.setDebug()
        func.setDebug()
        equation.setDebug()
        constraint.setDebug()
        arglist.setDebug()
        arg.setDebug()
        expression.setDebug()
        component.setDebug()
        inequalityOp.setDebug()
        struct.setDebug()

    with open(args.source, 'r') as src_file:
        src = src_file.read()
        src_file.close()
    # Convert '=' to Eq in piecewise function, otherwise sympy cannot parse correctly.
    src = COND_EQ.sub(lambda p: ', Eq(' + p.group(1) + ', ' + p.group(2) + '))', src)
    # Convert '.' expresion to custom name clone extension
    # because sympy cannot accept '.' in variable names.
    src = NAME_EXT.sub(lambda p: p.group(1) + Names.clone_ext + \
            Names.clone_ext.join(p.group(2).split('.')), src)
    parsed_toks = program.parseString(src)

if __name__ == '__main__':
    main()
