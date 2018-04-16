from helpers import SympyHelper

import copy
import numpy as np
import re
import logging
from sets import Set

class Parser(object):
    START_CHAR = ['[', '{']
    END_CHAR = [']', '}']
    EXT_OP = [',', ',']
    #PAT_EXT = [re.compile(r'\[.+\]'), re.compile(r'\{.+\}')]
    
    def __init__(self):
        assert len(self.START_CHAR) == len(self.END_CHAR)
        assert len(self.END_CHAR) == len(self.EXT_OP)
        self.pat_ext = [re.compile('\\'+sc+'.+\\'+ec)
                for sc, ec in zip(self.START_CHAR, self.END_CHAR)]
        self.ext_op_map = dict([(ec, op) for ec, op in zip(self.END_CHAR, self.EXT_OP)])

    def _gen_pat_idx(self, idx):
        pat_string = '\_' + idx + '$'
        return re.compile(pat_string)

    def _gen_pat_sub(self, syms, i):
        syms = np.array(syms, ndmin=1, copy=False)
        subs = [sym[:sym.rfind('_')] + '_' + str(i) for sym in syms]
        return subs[0] if len(subs) == 1 else subs

    def _gen_expr_by_sub(self, victims, subs, exprs, keep):
        assert len(victims) == len(subs), 'Parser -- substitution term numbers mismatch'
        new_exprs = Set(exprs)
        for victim, sub in zip(victims, subs):
            for expr in new_exprs:
                new_exprs.remove(expr)
                new_exprs.add(expr.replace(victim, sub))
        exprs = exprs + list(new_exprs) if keep else list(new_exprs)
        return exprs
    
    def _get_next_ext(self, expr):
        """ Returns the next most inner squred parenthesis.
        """
        stack = []
        for i, c in enumerate(expr):
            if c in self.START_CHAR:
                stack.append(i)
            elif c in self.END_CHAR:
                assert stack, 'Parser -- non-matching parenthesis'
                start = stack.pop()
                assert i > start
                return start, i+1, self.ext_op_map[c]
        return None, None, None

    def _gen_expr_by_ext(self, expr, idx_bounds, syms):
        # Get first inner sp.
        start, end, ext_op = self._get_next_ext(expr)
        while start and end:
            candidate_str = expr[start:end]
            # Get rid of '[' and ']' in sub strings.
            victim_str = candidate_str[1:-1]
            candidate_syms = []
            # Get all symbols inside victim_str.
            for k in syms:
                if k in victim_str:
                    candidate_syms.append(k)

            # Initialize replacement victim for every interval [start, end).
            sub_comps = [victim_str]
            # For every index.
            for idx in idx_bounds:
                pat = self._gen_pat_idx(idx)
                sub_str = {}
                # For all indexed symbol matching idx.
                for sym in candidate_syms:
                    if pat.search(sym):
                        tmp_syms = []
                        for i in xrange(*idx_bounds[idx]):
                            tmp_syms.append(self._gen_pat_sub(sym, i))
                        # sym should be replaced with tmp_syms
                        sub_str[sym] = tmp_syms
                if sub_str:
                    new_sub_comps = []
                    for comp in sub_comps:
                        for i in xrange(*idx_bounds[idx]):
                            new_comp = comp
                            # Replace all index_idx within comp.
                            for sym in sub_str:
                                new_comp = new_comp.replace(sym, sub_str[sym][i])
                            new_sub_comps.append(new_comp)
                    # Update substitution components list.
                    sub_comps = new_sub_comps
            # Add brackets around each component to guarentee ordering when reduced later.
            #sub_comps = ['('+comp+')' for comp in sub_comps]
            # Generate sum string.
            rep_str = reduce(lambda x,y: x+' '+ext_op+' '+y, sub_comps)
            expr = expr.replace(candidate_str, rep_str)
            start, end, ext_op = self._get_next_ext(expr)
        return expr

    def expand_syms(self, idx_bounds, symbol_dict):
        expanded_syms = symbol_dict
        for idx in idx_bounds:
            pat = self._gen_pat_idx(idx)
            idx_syms = [k for k in symbol_dict if pat.search(k)]
            for i in xrange(*idx_bounds[idx]):
                new_syms = self._gen_pat_sub(idx_syms, i)
                expanded_syms.update(SympyHelper.initSyms(new_syms))
        return expanded_syms

    def expand(self, exprs, idx_bounds, expanded_syms):
        logging.debug('Parser -- Idx bounds:\n\t{}'.format(idx_bounds))
        logging.debug('Parser -- exprs:\n\t{}'.format('\n\t'.join(exprs)))
        expanded_exprs = copy.deepcopy(exprs)
        # For nested indexed symbols.
        for expr in exprs:
            if any([pat.search(expr) for pat in self.pat_ext]):
                ext_expr = self._gen_expr_by_ext(expr, idx_bounds, expanded_syms)
                expanded_exprs.remove(expr)
                expanded_exprs.append(ext_expr)

        logging.debug('Parser -- ext exprs:\n\t{}'.format('\n\t'.join(expanded_exprs)))
        # For top-level indexed symbols.
        for idx in idx_bounds:
            pat = self._gen_pat_idx(idx)
            # Find all indexed symbols.
            idx_syms = [k for k in expanded_syms if pat.search(k)]
            (lower, upper) = idx_bounds[idx]
            for i in xrange(lower, upper):
                # Generate replacements.
                sub_strs = self._gen_pat_sub(idx_syms, i)
                # Expand equations with substitution.
                expanded_exprs = self._gen_expr_by_sub(idx_syms,
                        sub_strs, expanded_exprs, keep=True if i < upper-1 else False)
        logging.debug('Parser -- expanded exprs:\n\t{}'.format('\n\t'.join(expanded_exprs)))
        return SympyHelper.initExprs(expanded_exprs, expanded_syms)
