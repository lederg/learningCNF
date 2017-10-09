#!/usr/bin/env python
#
# USAGE: ./randomCNF <number_of_ground_variables> <number_of_files_to_generate>
#

import os, sys
from subprocess import Popen, PIPE, STDOUT

def is_number(s):
    try:
        int(s)
    except ValueError:
        return False
    return True

def randomCNF():
    fraction_of_additional_clauses = int(55 - 2.3 * int(sys.argv[1]))
    fuzz = Popen("./fuzzsat-0.1/fuzzsat -i {} -I {} -p {} -P {}".format(sys.argv[1],sys.argv[1],fraction_of_additional_clauses,fraction_of_additional_clauses), shell=True, stdout=PIPE, stderr=STDOUT)
    # fuzz = subprocess.Popen("./fuzzsat-0.1/fuzzsat -i 3 -I 500 -p 10 -P 20", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return fuzz.stdout.readlines()

def normalizeCNF(cnf):
    occs = {}
    maxvar = None

    for line in cnf:
        lits = line.split()[0:-1]
        if is_number(lits[0]):
            lits = list(map(int,lits))
            for l in lits:
                if abs(l) not in occs:
                    occs[abs(l)] = []
                occs[abs(l)] += [lits] # storing reference to lits so we can manipulate them consistently
        else:
            if lits[0] == b'p':
                maxvar = int(lits[2])
            
    assert(maxvar != None)

    # define sign, as we need it in the next for loop
    sign = lambda x: x and (1, -1)[x < 0]

    # Split variables when they occur in more than 8 clauses
    itervars = set(occs.keys())
    added_vars = 0
    while True:
        if len(itervars) == 0:
            break
        v = itervars.pop()
        if len(occs[v]) > 8:
            # print('Found var '+str(v)+ ' with '+ str(len(occs[v]))+ ' occurrences.')
            maxvar += 1
            added_vars += 1
            connector_clauses = [[v,-maxvar],[-v,maxvar]]
            ## prepend connector_clauses to shift all clauses back, don't want to remove the connector_clauses with what follows
            occs[v] = connector_clauses + occs[v] 
            assert(len(occs[v][10:]) > 0) # > 8 and we added the two connector_clauses
        
            assert(maxvar not in occs)
            occs[maxvar] = connector_clauses
        
            # move surplus clauses over to new variable
            for clause in occs[v][8:]:
                # change clause inplace, so change is consistent for occurrence lists of other variables
                clause[:] = list(map(lambda x: maxvar * sign(x) if abs(x) == v else x, clause))
                occs[maxvar] += [clause]
            assert(len(occs[v]) > len(occs[maxvar]))
        
            occs[v] = occs[v][:7]
        
            # if len(occs[maxvar]) > 8: 
                # print('  new var '+str(maxvar)+ ' will be back.')
        
            itervars.add(maxvar)
            
    # print ('  maxvar: ' + str(maxvar))
    # print ('  added vars: ' + str(added_vars))
    # print ('  Max: ' + str( max( [len(occs[v]) for v in occs.keys()] )))
    # print ('  Over 8 occs: ' + str(len( filter(lambda x: x, [len(occs[v]) > 8 for v in occs.keys()] ))))

    return maxvar, occs


def occs_to_clauses(maxvar, occs):
    clauses = set()
    clause_list = []
    for v in occs.keys():
        for c in occs[v]:
            c_string = ' '.join(map(str,c))
            if c_string not in clauses:
                clauses.add(c_string)
                clause_list.append(c_string)
    return clause_list



def write_to_file(maxvar, clause_list, filename):
    textfile = open(filename, "w")
    textfile.write('p cnf {} {}\n'.format(maxvar,len(clause_list)))
    # textfile.write('p cnf ' + str(maxvar) + ' ' + str(len(clause_list)) + '\n')
    for c in clause_list:
        textfile.write(c + ' 0\n')
    textfile.close()

    # sat2 = subprocess.Popen("picosat {}".format(filename),
    #                         shell=True,
    #                         stdout=subprocess.PIPE,
    #                         stderr=subprocess.STDOUT)
    # sat2.communicate()[0]
    # print('  ' + str(sat2.returncode))



def is_sat(maxvar,clause_list):
    p = Popen(['picosat'],stdout=PIPE,stdin=PIPE)
    
    p.stdin.write(str.encode('p cnf {} {}\n'.format(maxvar,len(clause_list))))
    for c in clause_list:
        p.stdin.write(str.encode(c + ' 0\n'))
    
    p.communicate()[0]
    # print('  ' + str(p.returncode))
    p.stdin.close()
    return p.returncode == 10


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
        
        cnfstring = randomCNF()
        maxvar, occs = normalizeCNF(cnfstring)
        clauses = occs_to_clauses(maxvar, occs)
        
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
