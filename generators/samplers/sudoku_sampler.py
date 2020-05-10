"""
SUDOKU (NUMBER PLACE) PUZZLE GENERATOR
by Arel Cordero November 12, 2005

This program is released into the public domain.
Revision 3
"""
import os
import time
import pickle
import random, copy
import numpy as np

from gen_types import FileName
from samplers.sampler_base import SamplerBase
from gen_utils import random_string
from pysat.solvers import Glucose3

import itertools

def negate(literal):
    return literal*(-1)

def rcn2var(size,row,col,num):
    """Return the variable which represents a (row,col,num)."""
    var = row * size**2 + col * size + num + 1 # +1 to avoid 0    
    return int(var)

def var2rcn(size, var):
    var = int(abs(var))-1
    res = int(var % size**2)
    row = int((var - res) / size**2)
    num = res % size
    col = int((res - num) / size)
    return row, col, num

class CNF_formula:
    def __init__(self,encoding_nvars):
        self.clauses = []
        self.nvars = encoding_nvars

    def print_dimacs(self,fname = None):
        """Write the cnf formula with dimacs format
        in the fname file.
        """
        if fname == None:
            import sys
            f  = sys.stdout
        else:
            f = open(fname,'w')

        def print_dimacs_clause(clause):
            f.write(" ".join(map(str,clause))+" 0\n")

        f.write("c\nc CNF Sudoku Translation\nc\np cnf "+str(self.nvars)+" "
                +str(len(self.clauses))+"\n")
        for clause in self.clauses:
            print_dimacs_clause(clause)

        if fname != None: f.close()

    def num_vars(self):
        return self.nvars

    def get_clauses(self):
        return self.clauses

    def implies(self,a,b):
        self.clauses.append([a*(-1),b])

    def biimplies(self,a,b):
        self.implies(a,b)
        self.implies(b,a)

    def at_least_one(self,B):
        self.clauses.append(B)

    def at_most_one(self,B):
        for combination in itertools.combinations(range(len(B)),2):
            self.clauses.append([(-1)*B[combination[0]],(-1)*B[combination[1]]])

    def addclause(self,cl):
        self.clauses.append(cl)

    def exacly_one(self,B, classic=True):
        """ Exacly one restriction.
        """

        if classic:
            self.at_least_one(B)
            self.at_most_one(B)

        else:
            n = len(B)
            R = list(range(self.nvars,self.nvars + n))
            R[0]=None
            print('Increasing nvars by {} to {}'.format(n-1,self.nvars))
            self.nvars += (n - 1)
            #Rn -> Rn-1
            for i in range(2,n):self.implies(R[i],R[i-1])
            #B1 <-> no(R2)
            self.biimplies(B[0],negate(R[1]))
            #Bn <-> Rn
            self.biimplies(B[-1],R[-1])
            #Bi <-> Ri and no(Ri+1)
            for i in range(1,n - 1):
                self.clauses.extend([[negate(B[i]),R[i]],
                            [negate(B[i]), negate(R[i+1])],
                            [negate(R[i]),R[i+1],B[i]]])

class SudokuCNF:
    def __init__(self, order, puzzle, classic=False):
        self.order = order
        self.size  = order**2
        self.sudoku = puzzle
        self.classic = classic
        self.var2rcn = {}

    def variable(self,row,col,num):
        """Return the variable which represents a (row,col,num)."""
        var = row * self.size**2 + col * self.size + num + 1 # +1 to avoid 0
        self.var2rcn[var] = (row, col, num)
        return var

    def as_features(self, encoding='binary'):
        assert encoding=='binary'
        puz = self.sudoku.copy()
        puz[puz==-1]=self.size
        return np.stack([puz==i for i in range(self.size+1)], axis=2)

    def save_annotation(self, fname):
        with open(fname, 'wb+') as f:
            pickle.dump(self.sudoku,f)

    def extended_translate_to_CNF(self):
        """ Translate to CNF all restrictions with all variales"""

        cnff = CNF_formula(self.size**3)
        for i in range(self.size):
            for j in range(self.size):
                A = [  self.variable(i,j,k) for k in range(self.size)]
                cnff.exacly_one(A, self.classic)# Cell restrictions
                B = [  self.variable(i,k,j) for k in range(self.size)]
                cnff.exacly_one(B, self.classic)# Row restrictions
                C = [  self.variable(k,i,j) for k in range(self.size)]
                cnff.exacly_one(C, self.classic)# Col restrictions
                if self.sudoku[i][j] != -1: #Fixed variables restrictions
                    cnff.addclause([self.variable(i,j,int(self.sudoku[i][j]))])

        #Region restrictions
        for (block_i, block_j) in itertools.product(range(self.order), range(self.order)):
            i, j = block_i * self.order, block_j * self.order
            for num in range(self.size):
                D = [ self.variable(i+i_inc,j+j_inc,num)  for i_inc in range(self.order)
                      for j_inc in range(self.order) ]
                cnff.exacly_one(D, self.classic)

        return cnff,[i for i in range((self.size**3)+1)]

    def display(self):
        for row in self.sudoku:
            print(' '.join([f'{n+1:>2}' if n+1 else '--' for n in row]))

    def encode(self):
        cnf, decoding_map = self.extended_translate_to_CNF()

        return cnf

    def fit(self, model):
        decoded = np.full((self.size, self.size), -1, dtype=int)
        for var in range(self.size**3):
            if (model[var] < 0): continue # The variable is False

            row, col, num = self.var2rcn[model[var]]
            assert(decoded[row][col] == -1), f"ERROR: Reassigning puzzle[{row}][{col}] = {decoded[row][col]} to {num}"
            decoded[row][col] = num

        return SudokuCNF(self.order, decoded, classic=self.classic)

    def pluck(self, count):
        ind = np.hstack(([0] * count, [1] * (self.size**2 - count)))
        np.random.shuffle(ind)
        mask = ind.reshape((self.size, self.size))
        puzzle = mask * (self.sudoku + 1) - 1
        return SudokuCNF(self.order, puzzle, classic=self.classic)

class SudokuSampler(SamplerBase):
    def __init__(self, order=3, num_filled=None, seed=None, use_classic=True, **kwargs):
        SamplerBase.__init__(self, **kwargs)
        if seed is None:
            seed = int(time.time())
        else:
            seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.use_classic = use_classic
        self.order = int(order)
        self.size = self.order**2
        self.num_filled = int(num_filled) if (num_filled) else self.size

    def rand_puzzle(self):
        while(True):
            seed_puzzle = np.hstack((np.random.choice(self.size, self.order), [-1] * (self.size**2 - self.order)))
            np.random.shuffle(seed_puzzle)
            seed_puzzle = seed_puzzle.reshape((self.size, self.size))
            sudokuCNF = SudokuCNF(self.order, seed_puzzle, classic=self.use_classic)
            cnf = sudokuCNF.encode()

            glucose = Glucose3(gc_freq="fixed")
            glucose.append_formula(cnf.get_clauses())
            if (not glucose.solve()): # The seed_puzzle is not solvable. Try again!
                continue

            ######## UNCOMMENT FOR TESTING ########
            # fitted = sudokuCNF.fit(glucose.get_model())
            # plucked = fitted.pluck(self.size**2 - self.num_filled)

            # sudokuCNF.display()
            # print("--------")
            # fitted.display()
            # print("--------")
            # plucked.display()
            # print("========")
            ######## END OF TESTING ########
            return sudokuCNF.fit(glucose.get_model()).pluck(self.size**2 - self.num_filled)


    def sample(self, stats_dict: dict) -> (FileName, FileName):
        cnf_id = "sudoku-%ix%i-%i-%s" % (self.size, self.size, self.num_filled, random_string(8))
        fname = os.path.join("/tmp", f"{cnf_id}.cnf")
        puzzle = self.rand_puzzle()
        puzzle.encode().print_dimacs(fname)
        stats_dict.update({
            'file': cnf_id,
            'order': self.order,
            'num_filled': self.num_filled
        })

        self.log.info(f"Sampled {cnf_id}")

        return fname, None
