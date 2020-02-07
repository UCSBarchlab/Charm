import collections

# Workaround for Sympy not lambdifying Product correctly, ref:
# https://stackoverflow.com/questions/37846492/sympy-cannot-lambdify-product
import sympy.printing.lambdarepr as SPL
import z3
from sympy import symbols, IndexedBase, Function
from sympy.parsing.sympy_parser import parse_expr, convert_equals_signs, auto_symbol


def _print_Product(self, expr):
    loops = (
        'for {i} in range({a}, {b}+1)'.format(
            i=self._print(i),
            a=self._print(a),
            b=self._print(b))
        for i, a, b in expr.limits)
    return '(prod([{function} {loops}]))'.format(
        function=self._print(expr.function),
        loops=' '.join(loops))
SPL.NumPyPrinter._print_Product = _print_Product

class Z3Helper(object):
    @staticmethod
    def sym2real(x):
        dynamic_src = str(x) + ' = z3.Real("' + str(x) + '")'
        return compile(dynamic_src, '', 'exec')

    @staticmethod
    def sym2bool(x):
        dynamic_src = str(x) + ' = z3.Bool("' + str(x) + '")'
        return compile(dynamic_src, '', 'exec')

    @staticmethod
    def getIterable(x):
        if isinstance(x, collections.Iterable):
            return x
        else:
            return (x,)

    # input args: 
    #   solver:  solver ref
    #   syms:    symbols in expr
    #   exprs:   the expressions to add as constraints
    @staticmethod
    def addCons(solver, syms, exprs):
        for sym in Z3Helper.getIterable(syms):
            exec(Z3Helper.sym2real(sym))
        expr_strs = str(exprs).replace("sqrt", "z3.Sqrt")
        solver.add(eval(expr_strs))

    @staticmethod
    def addSoftCons(solver, ctrl, syms, exprs):
        for sym in Z3Helper.getIterable(syms):
            exec(Z3Helper.sym2real(sym))

        # convert ctrl to z3 bool
        exec(Z3Helper.sym2bool(ctrl))

        # add constraints one by one
        for expr in Z3Helper.getIterable(exprs):
            expr_str = str(expr).replace("sqrt", "z3.Sqrt")
            print(eval(expr_str))
            solver.add(z3.Implies(eval(str(ctrl)), eval(expr_str)))
        
        return {str(ctrl): eval(str(ctrl))}


class SympyHelper(object):

    @staticmethod
    def initSyms(syms, deferred=False):
        sym_dict = {}
        if not isinstance(syms, list) and not isinstance(syms, set):
            syms = [syms]
        for sym in syms:
            sym_dict[sym] = symbols(sym) if not deferred else IndexedBase(sym)
        return sym_dict

    @staticmethod
    def initFuncs(funcs):
        func_dict = {}
        for func in funcs:
            func_dict[func] = symbols(func, cls=Function)
        return func_dict

    @staticmethod
    def initExprs(exprs, syms):
        #def can_split(symbol):
        #    if '.' in symbol:
        #        return False
        #    return _token_splittable(symbol)

        expr_list = []
        for expr in exprs:
            print(expr)
            parsed_expr = parse_expr(expr, local_dict=syms,
                    transformations=(convert_equals_signs, auto_symbol))
            expr_list.append(parsed_expr)
        return expr_list
