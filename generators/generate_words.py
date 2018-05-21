#!/usr/bin/env python3

import os
import sys
import argparse
import inspect

from random import randint, seed
from aiger import bv, bv_utils
from cadet_cmdline_utils import eval_formula

word_length = 8

arith_ops = [bv.BV.__add__, bv.BV.__sub__]
bitwise_ops = [bv.BV.__and__, bv.BV.__or__, bv.BV.__xor__]
shift_ops = [bv.BV.__rshift__, bv.BV.__lshift__]
unary_ops = [bv.BV.reverse, bv.BV.__invert__, bv.BV.__abs__, bv.BV.__neg__]
cmp_ops = [bv.BV.__eq__, bv.BV.__ne__, bv.BV.__lt__, bv.BV.__le__, bv.BV.__gt__, bv.BV.__ge__]

leafs = [bv.BV(word_length, 'x'), bv.BV(word_length, 1)]

variables = ['1 x1', '1 x2', '2 y1', '2 y2']

def variable():
    v = bv.BV(word_length, variables[randint(0, len(variables) - 1)])
    return v

def constant_expr():
    if randint(0, 4) == 0:
        c = bv.BV(word_length, randint(- 2**word_length + 1, 2**word_length -1))
    else: 
        c = bv.BV(word_length, 1)
    return c

def leaf_expr(size):  # constant or variable
    assert size == 1
    return variable() if randint(0, 1) == 0 else constant_expr()

def shift_expr(size):
    assert size > 1
    arg = random_expr(size - 1)
    shift_by = randint(1, word_length - 1)
    return shift_ops[randint(0, len(shift_ops) - 1)](arg, shift_by)

def unary_expr(size):
    assert size > 1
    arg = random_expr(size - 1)
    op = unary_ops[randint(0, len(unary_ops) - 1)]
    return op(arg)

def arithmetic_expr(size):
    assert size > 2
    operator = arith_ops[randint(0, len(arith_ops)-1)]
    # arg_num = len(inspect.getargspec(operator)[0])  # arity of the function
    split = randint(1, size - 2)  # at least one operation on either side
    left = random_expr(split)
    right = random_expr(size - split - 1)
    return operator(left, right)

def random_expr(size):
    if size <= 1:
        return leaf_expr(size)
    if size == 2:
        if randint(0, 2) == 0:
            return shift_expr(size)
        else:
            return unary_expr(size)
    if randint(0, 2) == 0:
        return bitwise_expr(size)
    else:
        return arithmetic_expr(size)

def bitwise_expr(size):
    assert size > 2
    op = bitwise_ops[randint(0, len(bitwise_ops)-1)]
    split = randint(1, size - 2)  # at least one operation on either side
    left = random_expr(split)
    right = random_expr(size - split - 1)
    return op(left, right)

def random_bool_expr(size):
    assert size > 2
    op = cmp_ops[randint(0, len(cmp_ops)-1)]
    split = randint(1, size - 2)  # at least one operation on either side
    left = random_expr(split)
    right = random_expr(size - split - 1)
    return op(left, right)


def random_circuit(size):
    while True:
        e = random_bool_expr(size)
        e = bv_utils.simplify(e)
        if e is not None:
            return e


def write_circuit(c):
    print(f'File {i} Size {e.aig.header.num_ands}\n' + '\n'.join(e.aig.comments))
    f = open(f'data/words{i}.aag', 'w')
    f.write(str(e))
    f.close()
    i += 1


def parse_cmdline():
    print('')
    p = argparse.ArgumentParser()
    # p.add_argument('-u', '--universals', dest='universals_num', action='store',
    #                metavar='U', nargs='?', default=0, type=int,
    #                help='The maximal number of variables that are '
    #                'turned universal')
    p.add_argument('-s', '--seed', dest='seed', action='store',
                   nargs='?', default=None, type=int, metavar='S',
                   help='Seed for the PNG. Uses fresh seed every run per default.')
    p.add_argument('--max_hardness', dest='max_hardness', action='store',
                   nargs='?', default=300, type=int, metavar='H',
                   help='The maximal average number of decisions required '
                   'to solve the problem.')
    p.add_argument('--min_hardness', dest='min_hardness', action='store',
                   nargs='?', default=1, type=int, metavar='h',
                   help='The minimal average number of decisions required'
                   'to solve the problem.')
    p.add_argument('--maxvars', dest='maxvars', action='store',
                   nargs='?', default=50, type=int, metavar='V',
                   help='The maximal number of variables (default 50).')
    # p.add_argument('--maxclauses', dest='maxclauses', action='store',
                   # nargs='?', default=150, type=int, metavar='C',
                   # help='The maximal number of clauses (default 150).')
    p.add_argument('-n', '--number', dest='num_generated', action='store',
                   nargs='?', default=1, type=int, metavar='N',
                   help='Number of files to be generated.')
    p.add_argument('-r', '--repetitions', dest='repetitions', action='store',
                   nargs='?', default=1, type=int, metavar='R',
                   help='Number of runs of CADET to compute average decisions.')
    p.add_argument('-p', '--prefix', dest='file_prefix', action='store',
                   nargs='?', default='', type=str, metavar='P',
                   help='Prefix given to all files.')
    p.add_argument('-w', '--word_size', dest='word_size',
                   action='store', nargs='?', default=8, type=int, metavar='W',
                   help='Word size (default 8).')
    p.add_argument('-e', '--expr_size', dest='expr_size',
                   action='store', nargs='?', default=8, type=int, metavar='W',
                   help='Number of nodes in the syntax tree of the expressions (default 8).')
    p.add_argument('-d', '--directory', dest='directory', action='store',
                   default='../data/words',
                   help='Directory to write the formulas to.')
    return p.parse_args()

def log_parameters(args):
    filename = os.path.join(args.directory, 'README')
    textfile = open(filename, "w")
    textfile.write(str(sys.argv))
    textfile.write('\n')
    textfile.write(str(args))
    textfile.close()

def main():
    args = parse_cmdline()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    log_parameters(args)

    if args.seed is not None:
        seed(args.seed)

    global word_length
    word_length = args.word_size

    file_extension = 'qaiger'
    num_sat = 0
    num_unsat = 0
    num_generated = 0
    num_attempts = 0

    while num_generated < args.num_generated:
        if num_attempts == 0:
            print('Generating file no {}'.format(num_generated+1))
        num_attempts += 1

        e = random_circuit(args.expr_size)

        if len(e.aig.gates) == 0:
            print('    Too few variables')
            continue
        if len(e.aig.gates) > args.maxvars:
            print('    Too many variables')
            continue
        if '1 x1' not in e.variables and '1 x2' not in e.variables:
            print('    No universals')
            continue

        (returncode, _, decisions) = eval_formula(str(e), args.repetitions, args.max_hardness * 10)
        if returncode == 30:
            print('    Hit the decision limit')
            continue
        if returncode not in [10, 20, 30]:
            errfiledir = '{}/err{}_{}.{}'.format(args.directory,
                                                 str(num_generated),
                                                 returncode,
                                                 file_extension)
            print(f"Warning: unexpected return code: {returncode};"
                  "writing formula to {errfiledir} and ignoring it")
            textfile = open(errfiledir, "w")
            textfile.write(str(e))
            textfile.close()
            continue

        if args.max_hardness >= decisions >= args.min_hardness:
            if returncode == 10:
                result_string = 'SAT'
                num_sat += 1
            else:  # returncode == 20:
                result_string = 'UNSAT'
                num_unsat += 1
            print(f'    Found a good formula! Decisions {decisions}; {result_string}')

            filedir = '{}/{}{}_{}.{}'.format(
                        args.directory,
                        args.file_prefix,
                        str(num_generated),
                        result_string,
                        file_extension)

            textfile = open(filedir, "w")
            textfile.write(str(e))
            textfile.close()
            num_generated += 1
            num_attempts = 0
        else:
            print(f'    Not the right number of decisions: {decisions}')

    print('Generated {} SAT and {} UNSAT formulas'.format(
            str(num_sat),
            str(num_unsat)))


if __name__ == "__main__":
    main()
