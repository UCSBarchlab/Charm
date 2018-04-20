import collections
import re
from sympy import symbols, numbered_symbols, IndexedBase, Idx, Function, DeferredVector
from sympy.parsing.sympy_parser import parse_expr, _token_splittable, convert_equals_signs, split_symbols_custom, auto_symbol

# Workaround for Sympy not lambdifying Product correctly, ref:
# https://stackoverflow.com/questions/37846492/sympy-cannot-lambdify-product
import sympy.printing.lambdarepr as SPL
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
            parsed_expr = parse_expr(expr, local_dict=syms,
                    transformations=(convert_equals_signs, auto_symbol))
            expr_list.append(parsed_expr)
        return expr_list
