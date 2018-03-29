#!/usr/bin/env python3

"""
This script generates random SAT and QBF formulas. 
"""

import os
import sys
import argparse
from cnf_tools import *
from random import randint
import time


def randomCNF(args):
    fraction_of_additional_clauses = int(55 - 2.5 * args.ground_vars)
    cmdstring = "./fuzzsat-0.1/fuzzsat -i {} -I {} -p {} -P {}"
    fuzz = Popen(cmdstring.format(str(args.ground_vars),
                                  str(args.ground_vars),
                                  fraction_of_additional_clauses,
                                  fraction_of_additional_clauses),
                 shell=True, stdout=PIPE, stderr=STDOUT)
    return fuzz.stdout.readlines()


def randomQBF(args):
    fraction_of_additional_clauses = 10
    cmdstring = "./fuzzsat-0.1/fuzzsat -i {} -I {} -p {} -P {} -l 3 -L 8"
    fuzz = Popen(cmdstring.format(str(args.ground_vars),
                                  str(args.ground_vars),
                                  fraction_of_additional_clauses,
                                  fraction_of_additional_clauses),
                 shell=True, stdout=PIPE, stderr=STDOUT)
    return fuzz.stdout.readlines()


def parse_cmdline():
    print('')
    p = argparse.ArgumentParser()
    p.add_argument('-u', '--universals', dest='universals_num', action='store',
                   metavar='U', nargs='?', default=0, type=int,
                   help='The maximal number of variables that are '
                   'turned universal')
    p.add_argument('--max_hardness', dest='max_hardness', action='store',
                   nargs='?', default=30, type=int, metavar='H',
                   help='The maximal average number of decisions required '
                   'to solve the problem.')
    p.add_argument('--min_hardness', dest='min_hardness', action='store',
                   nargs='?', default=1, type=int, metavar='h',
                   help='The minimal average number of decisions required'
                   'to solve the problem.')
    p.add_argument('--maxvars', dest='maxvars', action='store',
                   nargs='?', default=50, type=int, metavar='V',
                   help='The maximal number of variables (default 50).')
    p.add_argument('--maxclauses', dest='maxclauses', action='store',
                   nargs='?', default=150, type=int, metavar='C',
                   help='The maximal number of clauses (default 150).')
    p.add_argument('-n', '--number', dest='num_generated', action='store',
                   nargs='?', default=1, type=int, metavar='N',
                   help='Number of files to be generated.')
    p.add_argument('--prefix', dest='file_prefix', action='store',
                   nargs='?', default='', type=str, metavar='P',
                   help='Prefix given to all files.')
    p.add_argument('-g', '--ground_variables', dest='ground_vars',
                   action='store', nargs='?', default=8, type=int, metavar='G',
                   help='Number of ground variables for the SAT generator.')
    p.add_argument('-d', '--directory', dest='directory', action='store',
                   default='data/',
                   help='Directory to write the formulas to.')
    return p.parse_args()


def main(argv):
    args = parse_cmdline()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    is_qbf = args.universals_num > 0
    file_extension = 'qdimacs' if is_qbf else 'dimacs'
    num_sat = 0
    num_unsat = 0
    num_generated = 0
    num_attempts = 0
    print(str(args.num_generated))
    while num_generated < args.num_generated:
        print('Generating file no {}'.format(num_generated+1))
        num_attempts += 1


        dimacs = randomQBF(args) if is_qbf else randomCNF(args)
        maxvar, clauses = dimacs_to_clauselist(dimacs)
        # maxvar, clauses = normalizeCNF(clauses)

        if maxvar > args.maxvars:
            print('Too many variables in generated CNF')
            continue
        if len(clauses) > args.maxclauses:
            print('Too many clauses in generated CNF')
            continue

        # randomly select n variables to be universals;
        # find a set of universals that provokes many conflicts
        print('  maxvar {}'.format(str(maxvar)))
        candidate_universals = 0
        candidate_decisions = 0
        candidate_returncode = 0
        if is_qbf:
            for n in range(1, args.universals_num):
                universals = set()
                # print('  Trying n={} universals'.format(n))
                for _ in range(n):
                    candidate = randint(1, maxvar)
                    universals.add(candidate)
                assert(len(universals) > 0)

                (returncode, _, decisions) = eval_formula(maxvar,
                                                       clauses,
                                                       universals)
                if returncode not in [10, 20]:
                    errfiledir = '{}/err{}_{}.{}'.format(args.directory,
                                                         str(num_generated),
                                                         result_string,
                                                         file_extension)
                    print('Warning: unexpected return code: {}; \
                           writing formula to {} and ignoring it'.
                          format(returncode,
                                 errfiledir))
                    write_to_file(
                        maxvar,
                        clauses,
                        errfiledir,
                        universals)
                    continue
                if candidate_universals is None \
                  or candidate_decisions < decisions:

                    candidate_universals = universals
                    candidate_decisions = decisions
                    candidate_returncode = returncode
        else:  # SAT formula
            universals = set()
            (returncode, _, decisions) = eval_formula(
                                        maxvar,
                                        clauses,
                                        universals)
            if returncode not in [10, 20]:
                errfiledir = '{}/err{}_{}.{}'.format(
                                args.directory,
                                str(num_generated),
                                result_string,
                                file_extension)
                print('Warning: unexpected return code: {}; \
                       writing formula to {} and ignoring it'.
                      format(returncode, errfiledir))
                write_to_file(maxvar, clauses, errfiledir, universals)
                continue
            candidate_universals = universals
            candidate_decisions = decisions
            candidate_returncode = returncode
        
        print('candidate_decisions {}'.format(candidate_decisions))
        if args.max_hardness >= candidate_decisions >= args.min_hardness:
            if candidate_returncode == 10:
                result_string = 'SAT'
                num_sat += 1
            else:  # candidate_returncode == 20:
                result_string = 'UNSAT'
                num_unsat += 1
            print('  best candidate has {} universals, is {}, and '
                  'takes {} decisions'.format(
                        len(candidate_universals),
                        result_string,
                        candidate_decisions))

            filedir = '{}/{}_{}.{}'.format(
                        args.directory,
                        str(num_generated),
                        result_string,
                        file_extension)

            write_to_file(
                maxvar,
                clauses,
                filedir,
                candidate_universals)
            num_generated += 1
        else:
            print('Failed to generate candidate universals: '
                  'number of decisions is {}, which is not in bounds [{},{}]'.
                  format(decisions, args.min_hardness, args.max_hardness))

    print('Generated {} SAT and {} UNSAT formulas'.format(
            str(num_sat),
            str(num_unsat)))


if __name__ == "__main__":
    main(sys.argv)
