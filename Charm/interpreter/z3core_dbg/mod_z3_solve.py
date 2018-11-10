import z3

# Slightly modified version of the solve function found in z3.py
def mod_solve(*args, **keywords):
    s = z3.Solver()
    s.set(**keywords)
    s.add(*args)
    if keywords.get('show', False):
        print(s)
    r = s.check()
    if r == z3.unsat:
        print("no solution")
        return None
    elif r == z3.unknown:
        print("failed to solve")
        try:
            print(s.model())
            return None
        except z3.Z3Exception:
            return None
    else:
        #print(s.model())
        return s.model()
