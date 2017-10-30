#!/usr/bin/env python3

import sys
from os import listdir
from os.path import isfile, join
from cnf_tools import *
import ipdb

def simplify_clause(c):
    s = set(c)
    for x in s:
        if -x in s:
            return None
    return list(s)

def add_clauses(CNF, clauses):
    for t in clauses:
        c = simplify_clause(t)        
        if c != None:
            CNF['clauses'] += [c]
            for l in c:
                v = abs(l)
                if v not in CNF['clauses_per_variable']:
                    CNF['clauses_per_variable'][v] = []
                CNF['clauses_per_variable'][v] += [c]

def dimacs_to_cnf(filename):
    
    CNF = {'topvar' : None, \
           'maxvar' : None, \
           'origvars': {},  \
           'auxvars': [],   \
           'clauses': [],   \
           'clauses_per_variable' : {}}
    
    with open(filename, 'r') as f:
        
        numclauses = None
        
        for line in f.readlines():
            words = line.split()
            if is_number(words[0]):
                lits = list(map(int,words[0:-1]))
                add_clauses(CNF,[lits])
                
                # Assigns first singleton clause as topvar
                if len(lits) == 1 and CNF['topvar'] == None:
                    CNF['topvar'] = lits[0]
                
            else:
                if type(words[0]) == bytes and words[0] == b'p' \
                    or  \
                   type(words[0]) == str and words[0] == 'p':
                    CNF['maxvar'] = int(words[2])
                    numclauses = int(words[3])
        
        # if numclauses != len(CNF['clauses']):
        #     print('WARNING: Number of clauses in file is inconsistent.')
        
        assert(CNF['maxvar'] != None)
        CNF['origvars'] = {i: i for i in range(1,CNF['maxvar']+1)}
    
    for v in CNF['clauses_per_variable'].keys():
        if len(CNF['clauses_per_variable'][v]) > MAX_CLAUSES_PER_VARIABLE:
            print('Error: too many clauses for variable ' + str(v))
            ipdb.set_trace()
            quit()

    return CNF


def load_class(directory):
    files = [join(directory, f) for f in listdir(directory)]
    return list(map(dimacs_to_cnf, files))


def main(argv):
    
    print(load_class('data/randomCNF_5/sat/'))
    # eq_classes = {}
    # for eq_class_data in classes:
    #     eq_class = []
    #
    # eq_classes[formula_node['Symbol']] = eq_class
if __name__ == "__main__":
    main(sys.argv)
    