import logging
import z3
from abstract_syntax_tree import IdObject

class SMTInstance(IdObject):
    kSKIP = ['<', '>']

    def __init__(self, types, cons):
        assert isinstance(types, dict), 'Types for SMT must be a dictionary'
        assert isinstance(cons, list), 'Constraints for SMT must be a list'
        self.types = types
        self.cons = cons
        self.asmpts = []
        # Some may call this the dark side but
        # put SMT variables in object locals, so that we can refer to them
        # at other places.
        for v in self.types.keys():
            assert v not in vars(self)
            dym_str = 'vars(self)["{}"] = z3.{}("{}")'.format(
                    v, 'Real' if self.types[v] == 'float' else 'Int', v)
            exec(dym_str)
        goal = z3.ParThen('simplify', 'qfnra-nlsat')
        self.solver = goal.solver()
        for con in self.cons:
            trans_con = self.__transform(con)
            self.solver.add(trans_con)

    def __transform(self, con):
        # Create local variables that bind with SMT variables, so that we 
        # do not need to convert reference in constraints to object variables.
        for v in self.types.keys():
            dym_str = '{} = self.{}'.format(v, v)
            exec(dym_str)
        # TODO: dirty hack for CNN, fix this.
        if 'ceiling' in con:
            con = 'computation = (2*M*N) / '\
                    '(z3.If(M/T_m < z3.ToReal(M)/T_m, M/T_m + 1, M/T_m) * '\
                    'z3.If(N/T_n < z3.ToReal(N)/T_n, N/T_n + 1, N/T_n))'
        mutable = list(con)
        for i in xrange(len(mutable)):
            if con[i] == '=':
                mutable[i] = con[i] if con[i-1] in self.kSKIP or con[i+1] in self.kSKIP else '=='
        return eval(''.join(mutable))

    def dump(self):
        logging.debug('SMT types: {}'.format(self.types))
        logging.debug('SMT cons: {}'.format(self.cons))

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
            return None
        else:
            assert res == z3.unknown
            raise ValueError('z3 fails, aborting...')
