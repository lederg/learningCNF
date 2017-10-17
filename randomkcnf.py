#!/usr/bin/env python3
#
# USAGE: ./randomkcnf <number_of_variables> <number_of_files_to_generate>
#

from cnf_tools import *

import sys, os
from random import randint

def randomClause(maxVar, size):
    assert(size < maxVar)
    clause = set()
    while len(clause) < size:
        new = randint(1, maxVar)
        if new not in clause:
            polarity = 2 * randint(0,1) - 1
            clause.add(new * polarity)
    # print('  generated clause: ' + clause_to_string(list(clause)))
            
    return list(clause)
    
    # from numpy.random import choice
    # variables = choice(range(1,maxVar+1),size,False)
    # polarities = choice(range(0,2),size,False)
    # variables * polarities

def random3CNF(maxVar, numClauses):
    # for _ in range(numClauses):
    #     clauses += randomClause(maxVar, 3)
    # return clauses
    return [randomClause(maxVar, 3) for _ in range(numClauses)]


def main(argv):
    
    assert(len(sys.argv) == 3)
    assert(is_number(sys.argv[1]))
    assert(is_number(sys.argv[2]))
    
    if not os.path.exists('data/'):
        os.makedirs('data/')
    directory = 'data/random3CNF_{}/'.format(sys.argv[1])
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory+'sat/'):
        os.makedirs(directory+'sat/')
    if not os.path.exists(directory+'unsat/'):
        os.makedirs(directory+'unsat/')
    
    num_sat = 0
    num_unsat = 0
    
    for i in range(int(sys.argv[2])):
        # if i % 10 == 9:
        print('Generating file {}'.format(i + 1))
        
        numVars = int(sys.argv[1])
        numClauses = int( 4.258 * numVars + 58.26 * numVars**-.66666 ) # taken from paper: "Predicting Satisfiability at the Phase Transition" attributing it to Crawford and Auton 
        
        clauses = random3CNF(numVars,numClauses)
        # maxvar, clauses = normalizeCNF(clauses)
        maxvar = numVars
        
        if is_sat(maxvar,clauses):
            num_sat += 1
            write_to_file(maxvar, clauses, '{}/sat/sat-{}.cnf'.format(directory,num_sat))
        else:
            num_unsat += 1
            write_to_file(maxvar, clauses, '{}/unsat/unsat-{}.cnf'.format(directory,num_unsat))
        
        # textfile = open("tmp.dimacs", "w")
        # textfile.writelines(cnfstring)
        # textfile.close()
        #
        # sat = subprocess.Popen("picosat tmp.dimacs", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # sat.communicate()[0]
        # print(sat.returncode)

if __name__ == "__main__":
    main(sys.argv)
