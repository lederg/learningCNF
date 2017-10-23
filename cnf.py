#!/usr/bin/env python3
#
# USAGE: ./cnf <number_of_ground_variables> <number_of_files_to_generate>
#

import os, sys
from cnf_tools import *

def randomCNF():
    fraction_of_additional_clauses = int(55 - 2.3 * int(sys.argv[1]))
    fuzz = Popen("./fuzzsat-0.1/fuzzsat -i {} -I {} -p {} -P {}".format(sys.argv[1],sys.argv[1],fraction_of_additional_clauses,fraction_of_additional_clauses), shell=True, stdout=PIPE, stderr=STDOUT)
    # fuzz = subprocess.Popen("./fuzzsat-0.1/fuzzsat -i 3 -I 500 -p 10 -P 20", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return fuzz.stdout.readlines()


def main(argv):
    
    assert(len(sys.argv) == 3)
    assert(is_number(sys.argv[1]))
    assert(is_number(sys.argv[2]))
    
    
    if not os.path.exists('data/'):
        os.makedirs('data/')
    directory = 'data/randomCNF_{}/'.format(sys.argv[1])
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory+'sat/'):
        os.makedirs(directory+'sat/')
    if not os.path.exists(directory+'unsat/'):
        os.makedirs(directory+'unsat/')
    
    num_sat = 0
    num_unsat = 0
    
    for i in range(int(sys.argv[2])):
        # if i % 10 == 0:
        print('Generating file no {}'.format(i))
        
        clauses = dimacs_to_clauselist(randomCNF())
        maxvar, clauses = normalizeCNF(clauses)
        
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
