### Assumptions ###
# True: incoming edge to variable node
# False: outgoing edge from variable node

### Function Specification ###
# SAT problem solved with the use of Z3
# args = inputs: dictionary, equations: {int(eq_id), list of eq vars}
# the function returns a dict {eq, outgoing edges}

### Notes ###
# Assertions can be used to verify that all inputs have been provided by the user

import z3
import logging
from timeit import default_timer as timer

def graph_transform_z3(inputs, equations):

	### Variables and SAT Equations Declaration ###
	for inp in inputs:
        	exec("%s = z3.Bool('%s')" % (inp, inp))
		exec("%s_eq = z3.Bool('%s_eq')" % (inp, inp))

	for eq, eq_vars in equations.iteritems():
		exec("e%d = z3.Bool('e%d')" % (eq, eq))
		for var in eq_vars:
			exec("e%d%s = z3.Bool('e%d%s')" % (eq, var, eq, var))
          

	### Inputs ###
	string_inputs = ""
	for inp, value in inputs.iteritems():
        	string_inputs += "%s == %s, " % (inp, str(value))
	exec("problem = [%s]" % (string_inputs[:-2]))


	### Equations ###
	list_of_sat_eqs = []
	string_eqs = ""

	#*** SAT equations for system variables ***#
	dict_var_edges = {} # Dictionary of edges per variable
	for eq, eq_vars in equations.iteritems():
		for var in eq_vars:
			if (var) not in dict_var_edges:
                                dict_var_edges[var] = []
                                dict_var_edges[var].append(var)
                        dict_var_edges[var].append("e%d%s" % (eq, var))
	

	# Variables can have 0 or 1 incoming edges
	for var, list_of_var_edges in dict_var_edges.iteritems():
        	pntr = 0
        	string_var_edges = ""
        	for i in range(len(list_of_var_edges)):
                	string_var_edges += "z3.And("
                	for pos, edge in enumerate(list_of_var_edges):
                        	if pos == pntr:
                                	string_var_edges += "%s, " % (edge) 
                        	else:
                                	string_var_edges += "z3.Not(%s), " % (edge)
                	string_var_edges = "%s), " % (string_var_edges[:-2]) 
                	pntr += 1
		list_of_sat_eqs.append("%s_eq" % (var))
        	string_eqs += "%s_eq == z3.Or(%s), " % (var, string_var_edges[:-2]) 


	#*** System equations in SAT format ***#

	# Equations can have 0 or 1 outgoing edges
	for eq, eq_vars in equations.iteritems():
		pntr = 0
		string_eq_edges = ""
        	for i in range(len(eq_vars)+1):
                	string_eq_edges += "z3.And("
                	for pos, var in enumerate(eq_vars):
                        	if pos == pntr:
                                	string_eq_edges += "e%d%s, " % (eq, var)
                        	else:
                                	string_eq_edges += "z3.Not(e%d%s), " % (eq, var)
                	string_eq_edges = "%s), " % (string_eq_edges[:-2])
                	pntr += 1

		list_of_sat_eqs.append("e%d" % (eq))
		string_eqs += "e%d == z3.Or(%s), " % (eq, string_eq_edges[:-2])
	
 
	#*** Final SAT condition ***#
	string_all_sat_eqs = ""
	for sat_eq in list_of_sat_eqs:
        	string_all_sat_eqs += "%s, " % (sat_eq) 
	string_eqs += "z3.And(%s) == True" % (string_all_sat_eqs[:-2])

	exec("equations_SAT = [%s]" % (string_eqs))        

	#*** Return solution as a dict ***#
	solution_z3object, time = mod_solve(equations_SAT + problem)
	if solution_z3object:
                # Cannot convert solution_z3object to string directly because
                # it will get cut off when too long. Use the iterator instead.
                solution_z3 = {}
		solution_CHARM = {}
                for d in solution_z3object.decls():
                    solution_z3[d.name()] = solution_z3object[d]
		for eq, eq_vars in equations.iteritems():
			solution_CHARM[eq] = None
			for var in eq_vars:
                            name = "e%d%s" % (eq, var)
                            logging.debug(name + str(solution_z3[name]))
                            if z3.is_true(solution_z3[name]):
                                solution_CHARM[eq] = var
                                break
		return solution_CHARM, time
	else:
		return None, time


# Slightly modified version of the solve function found in z3.py
def mod_solve(*args, **keywords):
	s = z3.Solver()
    	s.set(**keywords)
    	s.add(*args)
    	if keywords.get('show', False):
        	print(s)
        start = timer()
    	r = s.check()
        end = timer()
        time = end - start
    	if r == z3.unsat:
        	print("Cannot convert to functional graph.")
		return None, time
    	elif r == z3.unknown:
        	print("Failed to convert to functional graph.")
        	try:
            		print(s.model())
			return None, time
        	except z3.Z3Exception:
            		return None, time
    	else:
        	return s.model(), time

def test():
    ### Example 1 - unsat ### 
    inputs = {'a': True, 'b': False, 'x': False, 'y': False}
    equations = {1: ['x', 'a', 'b'], 2: ['y', 'x', 'a']}
    #check_config(inputs, equations)
    print check_config(inputs, equations)

    ### Example 2 - sat ###
    inputs = {'a': True, 'b': False, 'x': True, 'y': False}
    equations = {1: ['x', 'a', 'b'], 2: ['y', 'x', 'a']}
    #check_config(inputs, equations)
    print check_config(inputs, equations)

#test()
