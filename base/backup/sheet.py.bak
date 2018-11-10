from helpers import *
from parser import Parser
from functools import partial
from sympy import *
from sympy.printing.latex import print_latex
from sympy.utilities.lambdify import lambdify, lambdastr
from scipy.optimize import minimize
from mcerp import *
from uncertainties import umath as a_umath
from uncertainties import wrap as uwrap
from uncertainties import ufloat
from utils.gaussian_decomposition import gaussian_decomposition
from utils.softmax import SoftMaximum
import numpy as np
import re
import logging
from copy import copy, deepcopy

class PredType(object):
    GIVEN = 1
    RESPONSE = 2

class Sheet(object):

    def __init__(self, analytical=False, tag=None):
        # Custom function mappings used for lamdify. TODO: deprecated. 
        # This should be handled by addFuncs now.
        #if analytical:
        #    self.sym2func = {"ceiling":a_umath.ceil, "Max":uwrap(SoftMaximum)}
        #    self.conv2analytical = self.conv2analytical_simple_compression
            #self.conv2analytical = self.conv2analytical_GMM
        #else:
        #    self.sym2func = {"ceiling":umath.ceil}
        self.tag = tag
        self.analytical = analytical
        self.sym2func = {}
        self.idx_bounds = {} # Bounds for index symbols, key type: string
        self.syms = {} # Symbolics used in system modeling, key type: string
        self.exprs_str = [] # Original string representation of expressions.
        self.exprs = [] # Sympy understandable parsed expressions.
        self.given = {} # Inputs to model evaluation, key type: symbol
        self.response = set() # set of symbols.
        self.ordered_given = [] # List of symbols.
        self.sol_set = {} # key type: symbol
        self.target_funcs = {} # key type: symbol
        self.opts = []
        self.parser = Parser()
        npts = 100

    def dump(self):
        print(self.exprs)

    def addSyms(self, sym_list):
        """ Add symbols.

        Args:
            sym_list: [string].
        """
        self.syms.update(SympyHelper.initSyms(sym_list))

    def addFuncs(self, func_dict):
        """ Add custom functions.
        """

        self.syms.update(SympyHelper.initFuncs(func_dict.keys()))
        self.sym2func.update(func_dict)

    def addExprs(self, expr_list):
        """Add equations in system.

        Args:
            expr_list: [string], all symbols mush have been defined with addSyms.
        """
        #self.exprs += SympyHelper.initExprs(expr_list, self.syms)
        self.exprs_str = expr_list

    def _predSanityCheck(self, t, predType):
        if predType is PredType.GIVEN:
            assert len(t) == 2 
            assert(isinstance(t[0], str) and (isinstance(t[1], float)
                or isinstance(t[1], UncertainFunction)))
        elif predType is PredType.RESPONSE:
            assert isinstance(t, str)
            t = [t]
        else:
            raise ValueError("pred type of %r not defined!" % t[0])
        if not t[0] in self.syms.keys():
            raise ValueError("%r not defined!" % t[0])

    def reset(self):
        self.given = {} # Inputs to model evaluation, key type: symbol
        self.response = set() # set of symbols.
        self.ordered_given = [] # List of symbols.
        self.sol_sets = {} # key type: symbol
        self.target_funcs = {} # key type: symbol
        self.opts = []

    def clear(self):
        self.response = set()

    # new values will overwirte old ones
    def addPreds(self, given=None, bounds=None, response=None):
        """ Add predicates.

        Args:
            given: [(var, value)]
            bounds: {k: (lower, upper)} 
            response: [string], var to solve for.
        """
        if bounds:
            self.idx_bounds = dict([(k, bounds[k]) for k in bounds])
            self.syms.update(self.parser.expand_syms(self.idx_bounds, self.syms))
       
        if given:
            for t in given:
                self.given[self.syms[t[0]]] = t[1]
        
        if response:
            for t in response:
                self._predSanityCheck(t, PredType.RESPONSE)
                self.response.add(self.syms[t])

    def conv2analytical_GMM(self, given):
        """ Converts MC given to a vector of Gaussians using GMM EM fitting.
        The conversion result of this function are a vector of KNOWN gaussians,
        so the collapsing with uncertainties package won't lose shape of the
        distribution at this point.
        """

        result = []
        for q in given:
            if isinstance(q, UncertainFunction):
                components = gaussian_decomposition(q)
                mix = 0
                for (pi, mu, sigma) in components:
                    mix += pi * ufloat(mu, sigma)
                logging.debug('Original Dist: {}, {}\nDecomposed Mix Dist: {}, {}'.format(
                    q.mean, (q.var)**.5, mix.n, mix.std_dev))
                result.append(mix)
            else:
                result.append(q)
        return result

    def conv2analytical_simple_compression(self, given):
        """
        Convertes MC given to analytical form compatible with uncertainties.
        """

        result = []
        for q in given:
            if isinstance(q, UncertainFunction):
                nominal = q.mean
                std = np.sqrt(q.var)
                result.append(ufloat(nominal, std))
            else:
                result.append(q)
        return result

    def optimize(self, ordered_given, q_ordered_given, maximize=False):
        """ Minimization on responses.

        Args:
            ordered_given: [var], free varibles in an ordered way,
                                  "constants" should be behind all optimization targets
            q_ordered_given: [float], values for "constant" free variables
                                      in the same ordred way as above
        Returns:
            opt_val: {var, opt_val}, dict holding opt_val for each optimizing var
        """

        sol_sets = solve(self.exprs, exclude=ordered_given, check=False, manual=True)[0]

        init_guesses = []
        opt_val = {}
        for k in self.opts:
            init_guesses.append(4)
            opt_val[k] = 4

        target_funcs = {}
        for var in self.response:
            if maximize:
                target_funcs[var] = lambdify(tuple(ordered_given), -1 * sol_sets[var])
            else:
                target_funcs[var] = lambdify(tuple(ordered_given), sol_sets[var])
            # TODO: parameterize bounds
            result = minimize(target_funcs[var], init_guesses,
                    args=tuple(q_ordered_given), bounds=[(0.1, 16.1)])
            if not result.success:
                print(result.message)
            else:
                for (k, v) in zip(self.opts, result.x):
                    opt_val[k] = v

        logging.debug("Sheet -- minimization: {}".format(opt_val))
        return opt_val

    def compute(self, maximize=False, constraints=None):
        """
        Solve the system and apply the quantification.
        """

        # Expand expressions on first time.
        if not self.exprs:
            self.exprs = self.parser.expand(self.exprs_str, self.idx_bounds, self.syms)

        u_math = umath if not self.analytical else a_umath

        # Generate an ordering for inputs.
        if not self.ordered_given:
            for (k, v) in self.given.iteritems():
                self.ordered_given.append(k)
        q_ordered_given = []

        # Ordered given list fed to optimization, might be different from ordered_given.
        opt_ordered_given = []
        opt_q_ordered_given = []

        self.opts = []
        for (k, v) in self.given.iteritems():
            if isinstance(v, str) and v == 'opt':
                self.opts.append(k)
            else:
                opt_ordered_given.append(k)
                opt_q_ordered_given.append(v)

        # Do minimization if needed.
        if self.opts:
            opt_given = []
            for k in self.opts:
                opt_given.append(k)
            opt_ordered_given = opt_given + opt_ordered_given
            opt_val = self.optimize(opt_ordered_given, opt_q_ordered_given, maximize)

        # Assemble q_ordered_given according to ordered_given.
        for k in self.ordered_given:
            if isinstance(self.given[k], str) and self.given[k] == 'opt':
                q_ordered_given.append(opt_val[k])
            else:
                q_ordered_given.append(self.given[k])
   
        # Solve for final solution set, use cached version if possible.
        if not self.sol_set:
            sol_sets = solve(self.exprs,
                    exclude=self.ordered_given, check=False, manual=True)
            assert len(sol_sets) == 1, 'Multiple solutios possible, consider rewrite model.'
            self.sol_set = sol_sets[0]
            logging.debug('Sheet -- Given: {}'.format(self.ordered_given))
            logging.debug('Sheet -- Solutions:')
            for k, s in self.sol_set.iteritems():
                logging.debug('\t{}: {}'.format(k, s))

        # Generate target funcs, use cached version if possible.
        for var in self.response:
            if var not in self.target_funcs:
                self.target_funcs[var] = (lambdify(tuple(self.ordered_given),
                    self.sol_set[var], modules=[self.sym2func, u_math]))
                logging.debug('Lamdification {} {} --\n\t{}'.format(var,
                    self.target_funcs[var],
                    lambdastr(tuple(self.ordered_given), self.sol_set[var])))
  
        # Compute response.
        q_response = {}
        for var in self.response:
            logging.debug('Solving {}'.format(str(var)))
            logging.debug('Params:\n{}\n{}'.format(
                self.ordered_given, q_ordered_given))
            logging.debug('Calling {}'.format(self.target_funcs[var]))
            perf = self.target_funcs[var](*tuple(q_ordered_given))
            q_response[str(var)] = perf

        return q_response
    
    def dprint(self, var):
        assert var in self.syms
        if self.syms[var] in self.sol_final_set:
            print('{}: {}'.format(var, self.sol_final_set[self.syms[var]]))
        else:
            print('{} does not have a solution yet, try print after evaluation.'.format(var))

    def printLatex(self):
        symbol_names = {}
        for var in self.given:
            symbol_names[var] = str(var)
        for expr in self.exprs:
            print(latex(expr, symbol_names=symbol_names))
        for var in self.response:
            print("{} = {}".format(str(var), latex(self.sol_final_set[var],
                symbol_names=symbol_names)))
            print("{} = {}".format(str(var), latex(self.sol_final_set[var])))
