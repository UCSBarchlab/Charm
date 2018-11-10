from sympy import *

class Constraint:
    # input args:
    #   raw_expr: sympy expression
    #   syms    : sympy symbol list
    def __init__(self, expr, syms):
        self.expr = expr
        self.syms = syms

    # add Gaussian error to a variable in the constraint
    def addVarErr(self, sym, sigma):
        self.sigmas[sym] = sigma
