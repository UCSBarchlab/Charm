import logging

import z3

from .abstract_syntax_tree import IdObject


class SMTInstance(IdObject):
    kSKIP = ['<', '>']

    def __init__(self, types, cons):
        super().__init__()
        assert isinstance(types, dict), 'Types for SMT must be a dictionary'
        assert isinstance(cons, list), 'Constraints for SMT must be a list'
        self.types = types
        self.cons = cons
        self.asmpts = []
        # Some may call this the dark side but
        # put SMT variables in object locals, so that we can refer to them
        # at other places.
        single = 'z3.Float32()'
        double = 'z3.Float64()'
        BV32 = 'z3.BitVecSort(32)'
        BV64 = 'z3.BitVecSort(64)'
        RNE = 'z3.RNE()'
        for v in list(self.types.keys()):
            assert v not in vars(self)
            if self.types[v] == 'float':
                dym_str = 'vars(self)["{}"] = z3.Const("{}", {})'.format(v, v, double)
            else:
                assert self.types[v] == 'int'
                dym_str = 'vars(self)["{}"] = z3.fpSignedToFP({}, z3.Const("{}", {}), {})'.format(v, RNE, v, BV64, double)
            exec(dym_str)
        # TODO Cannot find reference to Tactic, check for problems
        t1 = z3.Tactic('simplify')
        t2 = z3.Tactic('solve-eqs')
        t3 = z3.Tactic('split-clause')
        t4 = z3.Tactic('qffpbv')
        t5 = z3.Tactic('qfnra-nlsat')
        t6 = z3.Tactic('normalize-bounds')
        t7 = z3.Tactic('smt')
        goal = z3.Then(z3.AndThen(t1, t2, t6), t4)
        # goal = z3.AndThen(t1, t2, t6)
        self.solver = goal.solver()
        for con in self.cons:
            trans_con = self.__transform(con)
            self.solver.add(trans_con)

    def __transform(self, con):
        # Create local variables that bind with SMT variables, so that we 
        # do not need to convert reference in constraints to object variables.
        for v in list(self.types.keys()):
            dym_str = '{} = self.{}'.format(v, v)
            exec(dym_str)
        # TODO: dirty hack for CNN, fix this.
        if 'ceiling' in con:

            con = 'computation = (2*M*N) / '\
                    '(z3.fpRoundToIntegral(z3.RTP(), M/T_m) * '\
                    'z3.fpRoundToIntegral(z3.RTP(), N/T_n))'
        # example stmt: comp = (M * N) / (ceiling(M / T_m) * ceiling(N / T_n))
        # for occ in re.finditer('ceiling', con):
        #    c_body = ''
        #    stack = []
        #    assert con[occ.end()] == '('
        #    stack.append('(')
        #    for i in range(occ.end() + 1, len(con)):
        #        if con[i] == ')':
        #            stack.pop()
        #            if not stack:
        #                break
        #        elif con[i] == '(':
        #            stack.append('(')
        #        c_body += con[i]
        #    assert not stack
        #    replace_body = 'z3.If(z3.floor({}) < z3.ToReal(), z3.floor({}) + 1, z3.floor({})'.format(c_body)
        #    print replace_body

        mutable = list(con)
        for i in range(len(mutable)):
            if con[i] == '=':
                mutable[i] = con[i] if con[i - 1] in self.kSKIP or con[i + 1] in self.kSKIP else '=='
        return eval(''.join(mutable))

    def dump(self):
        logging.debug('{}'.format(self.solver.sexpr()))

    def makeAssumptions(self, asmpts):
        assert isinstance(asmpts, list), 'Assumptions for SMT must be a list'
        self.solver.push()
        for asmpt in asmpts:
            trans_asmpt = self.__transform(asmpt)
            self.solver.add(trans_asmpt)

    def clearAssumptions(self):
        self.solver.pop()

    def solve(self):
        res = self.solver.check()
        if res == z3.sat:
            m = self.solver.model()
            result = dict([(d.name(), m[d]) for d in m.decls()])
            return result
        elif res == z3.unsat:
            print('Unsatisfiable instance.')
            return None
        else:
            assert res == z3.unknown
            raise ValueError('z3 fails, aborting...')
