#!/usr/bin/env python3
#
# USAGE to generate SAT formulas: 
#   ./generateCNF.py sat <number_of_ground_variables> <number_of_files_to_generate> <target_number_of_conflicts>
#
# USAGE to generate QBF formulas: 
#   ./generateCNF.py qbf <number_of_ground_variables> <number_of_files_to_generate> <target_number_of_conflicts>
#
import os
import sys
from cnf_tools import *
from random import randint
import time

def randomCNF():
    ground_vars_num = sys.argv[2]
    fraction_of_additional_clauses = int(55 - 2.5 * int(ground_vars_num))
    fuzz = Popen("./fuzzsat-0.1/fuzzsat -i {} -I {} -p {} -P {}".format(ground_vars_num,ground_vars_num,fraction_of_additional_clauses,fraction_of_additional_clauses), shell=True, stdout=PIPE, stderr=STDOUT)
    # fuzz = subprocess.Popen("./fuzzsat-0.1/fuzzsat -i 3 -I 500 -p 10 -P 20", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return fuzz.stdout.readlines()

def randomQBF():
    ground_vars_num = sys.argv[2]
    fraction_of_additional_clauses = 10
    fuzz = Popen("./fuzzsat-0.1/fuzzsat -i {} -I {} -p {} -P {} -l 3 -L 8".format(ground_vars_num,ground_vars_num,fraction_of_additional_clauses,fraction_of_additional_clauses), shell=True, stdout=PIPE, stderr=STDOUT)
    # fuzz = subprocess.Popen("./fuzzsat-0.1/fuzzsat -i 3 -I 500 -p 10 -P 20", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return fuzz.stdout.readlines()

def main(argv):
    
    assert len(sys.argv) == 5
    assert sys.argv[1] == 'sat' or sys.argv[1] == 'qbf'
    assert(is_number(sys.argv[2]))
    assert(is_number(sys.argv[3]))
    assert(is_number(sys.argv[4]))
    
    target_number_of_conflicts = int(sys.argv[4])
    qbf = sys.argv[1] == 'qbf'
    
    if not os.path.exists('data/'):
        os.makedirs('data/')
    directory = 'data/random{}_{}_{:0.0f}/'.format('QBF' if qbf else 'SAT',sys.argv[2], time.time())
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_extension = 'qdimacs' if qbf else 'dimacs'
    
    num_sat = 0
    num_unsat = 0
    
    num_generated = 0
    while num_generated < int(sys.argv[3]):
        # if i % 10 == 0:
        print('Generating file no {}'.format(num_generated+1))
        
        maxvar, clauses = dimacs_to_clauselist(randomQBF() if qbf else randomCNF())
        # maxvar, clauses = normalizeCNF(clauses)
        # randomly select n variables to be universals; find a set of universals that provokes many conflicts
        print('  maxvar {}'.format(str(maxvar)))
        if qbf:
            for n in range(1,int(maxvar/8)):
                universals = set()
                # print('  Trying n={} universals'.format(n))
                for _ in range(n):
                    candidate = randint(1,maxvar)
                    universals.add(candidate)
                    # print(str(universals))
            
                assert((len(universals) > 0) == qbf)
                (returncode, conflicts) = eval_formula(maxvar,clauses,universals)
                if returncode not in [10,20]:
                    errfiledir = '{}/err{}_{}.{}'.format(directory, 
                                                         str(num_generated), 
                                                         result_string, 
                                                         file_extension)
                    print('Warning: unexpected return code: {}; \
                           writing formula to {} and ignoring it'.format(returncode, errfiledir))
                    write_to_file(
                        maxvar,
                        clauses,
                        errfiledir,
                        universals)
                    continue
                if candidate_universals == None or candidate_conflicts < conflicts:
                    candidate_universals = universals
                    candidate_conflicts = conflicts
                    candidate_returncode = returncode
        else: # SAT formula
            universals = set()
            (returncode, conflicts) = eval_formula(maxvar,clauses,universals)
            if returncode not in [10,20]:
                errfiledir = '{}/err{}_{}.{}'.format(directory, 
                                                     str(num_generated), 
                                                     result_string, 
                                                     file_extension)
                print('Warning: unexpected return code: {}; \
                       writing formula to {} and ignoring it'.format(returncode, errfiledir))
                write_to_file(
                    maxvar,
                    clauses,
                    errfiledir,
                    universals)
                continue
            candidate_universals = universals
            candidate_conflicts = conflicts
            candidate_returncode = returncode
            
        
        if candidate_conflicts != None and candidate_conflicts > randint(0,target_number_of_conflicts):
            if candidate_returncode == 10:
                result_string = 'SAT'
                num_sat += 1
            else: # candidate_returncode == 20:
                result_string = 'UNSAT'
                num_unsat += 1
            print('  best candidate has {} universals, is {}, and has {} conflicts'.format(
                        len(candidate_universals),
                        result_string,
                        candidate_conflicts))
            
            filedir = '{}/{}_{}.{}'.format(directory, str(num_generated), result_string, file_extension)
            
            write_to_file(
                maxvar,
                clauses,
                filedir,
                candidate_universals)
            num_generated += 1
        else:
            print('Failed to generate candidate universals: not enough conflicts')
            
        ### OLD SAT GENERATION CODE
        # else:
        #     assert sys.argv[1] == 'sat'
        #     if not os.path.exists(directory):
        #         os.makedirs(directory)
        #     if is_sat(maxvar,clauses):
        #         num_sat += 1
        #         result_string = 'SAT'
        #     else:
        #         num_unsat += 1
        #         result_string = 'UNSAT'
        #     print('  ' + result_string)
        #     num_generated += 1
        #     filedir = '{}/{}_{}.{}'.format(directory, str(num_generated), result_string, file_extension)
        #     write_to_file(
        #         maxvar,
        #         clauses,
        #         filedir)
        
        # textfile = open("tmp.dimacs", "w")
        # textfile.writelines(cnfstring)
        # textfile.close()
        #
        # sat = subprocess.Popen("picosat tmp.dimacs", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # sat.communicate()[0]
        # print(sat.returncode)
    
    print('Generated {} SAT and {} UNSAT formulas'.format(str(num_sat),str(num_unsat)))
    
if __name__ == "__main__":
    main(sys.argv)
