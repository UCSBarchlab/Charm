### Assumptions ###
# True: incoming edge
# False: outgoing edge

import z3
import mod_z3_solve

def check_config(inputs, equations, mem_exec = False):

    # Run from mem or file
    if mem_exec:
        solution_z3 = check_config_mem(inputs, equations)
    else:
        solution_z3 =check_config_file(inputs, equations)

    # Return dict {eq, outgoing edge}
    if solution_z3:
        solution_list = str(solution_z3)[1:-1].replace(' ','').split(',\n')
        solution_CHARM = {}
        for eq, eq_vars in equations.iteritems():
            for var in eq_vars:
                if "e%d%s=True" % (eq, var) in solution_list:
                    solution_CHARM[eq] = var 
            if (eq) not in solution_CHARM:
                solution_CHARM[eq] = None
        solution_CHARM
        return solution_CHARM
    else:
        return None

def check_config_file(inputs, equations):

    f = open("z3_sat_enc.py", "w+")
    f.write("import z3\n")
    f.write("import mod_z3_solve\n\n")
    f.write("def run_sat_from_file():\n")
    ### Variables and SAT Equations Declaration ###
    for inp in inputs:
        f.write("\t%s = z3.Bool('%s')\n" % (inp, inp))
        f.write("\t%s_eq = z3.Bool('%s_eq')\n" % (inp, inp))

    for eq, eq_vars in equations.iteritems():
        f.write("\te%d = z3.Bool('e%d')\n" % (eq, eq))
        for var in eq_vars:
            f.write("\te%d%s = z3.Bool('e%d%s')\n" % (eq, var, eq, var))

    ### Inputs ###
    string_inputs = ""
    for inp, value in inputs.iteritems():
        string_inputs += "\t\t%s == %s,\n" % (inp, str(value))
    f.write("\n\tproblem = [\n%s\n\t]\n" % (string_inputs[:-2]))

    ### Equations ###
    list_of_sat_eqs = []
    string_eqs = ""

    # SAT equations for system variables 
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
        string_eqs += "\t\t%s_eq == z3.Or(%s),\n" % (var, string_var_edges[:-2])

    # System equations in SAT format
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
        string_eqs += "\t\te%d == z3.Or(%s),\n" % (eq, string_eq_edges[:-2])

    # List of SAT equations
    string_all_sat_eqs = ""
    for sat_eq in list_of_sat_eqs:
        string_all_sat_eqs += "%s, " % (sat_eq)
    string_eqs += "\t\tz3.And(%s) == True\n" % (string_all_sat_eqs[:-2])
    f.write("\n\tequations_SAT = [\n%s\n\t]\n" % (string_eqs))

    ### Outro: Close file and run ###
    f.write("\n\t# Call Z3 solver\n")
    f.write("\treturn mod_z3_solve.mod_solve(equations_SAT + problem)")
    f.close()
    #os.system('python z3_sat_enc.py') 	
    ### Run Solver ###
    import z3_sat_enc
    return z3_sat_enc.run_sat_from_file()

def check_config_mem(inputs, equations):

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

    # SAT equations for system variables 
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

    # System equations in SAT format
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

    # List of SAT equations
    string_all_sat_eqs = ""
    for sat_eq in list_of_sat_eqs:
        string_all_sat_eqs += "%s, " % (sat_eq)
    string_eqs += "z3.And(%s) == True" % (string_all_sat_eqs[:-2])
    exec("equations_SAT = [%s]" % (string_eqs))


    ### Run Solver ###
    return mod_z3_solve.mod_solve(equations_SAT + problem)


### **** EXAMPLES **** ###
def test():
    inputs = {'a': False, 'c': False, 'b': False, 'e': False, 'd': False, 'f': False, 'g': False, 'i': False, 'h': False}
    equations = {9: ['i', 'e', 'a'], 13: ['h', 'b']}
    print check_config(inputs, equations, mem_exec = False)

test()
