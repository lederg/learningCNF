from cnf_parser import *
from utils import *
import numpy as np
from functools import partial
from torch.utils.data import Dataset
from enum import Enum
import time
import torch
import re
import collections
import os
import random
from IPython.core.debugger import Tracer
import ipdb

_use_shared_memory = False

MAX_VARIABLES = 1000000
MAX_CLAUSES = 5000000
# MAX_VARIABLES = 10000
# MAX_CLAUSES = 50000
GROUND_DIM = 8          # config.ground_dim duplicates this. 
IDX_VAR_UNIVERSAL = 0
IDX_VAR_EXISTENTIAL = 1
# IDX_VAR_MISSING = 2
IDX_VAR_DETERMINIZED = 2
IDX_VAR_ACTIVITY = 3
IDX_VAR_POLARITY_POS = 4
IDX_VAR_POLARITY_NEG = 5
IDX_VAR_SET_POS = 6
IDX_VAR_SET_NEG = 7

# external utility function to filter small formulas

def filter_dir(dirname, bound):
    a = [join(dirname, f) for f in listdir(dirname)]    
    rc = []
    for fname in a:
        if fname.endswith('qdimacs'):
            with open(fname,'r') as f:
                l = int(f.readline().split()[2])
                if l <= bound: rc.append(fname)
    return rc


class QbfBase(object):    
    def __init__(self, qcnf = None, **kwargs):
        self.sparse = kwargs['sparse'] if 'sparse' in kwargs else True
        self.qcnf = qcnf
        self.sp_indices = None
        self.sp_vals = None
        self.extra_clauses = {}
        self.removed_old_clauses = []
        # self.get_base_embeddings = lru_cache(max_size=16)(self.get_base_embeddings)
        if 'max_variables' in kwargs:
            self._max_vars = kwargs['max_variables']
        if 'max_clauses' in kwargs:
            self._max_clauses = kwargs['max_clauses']

    def reset(self):
        self.sp_indices = None
        self.sp_vals = None
        self.extra_clauses = {}
        

    def reload_qdimacs(self, fname):
        self.qcnf = qdimacs_to_cnf(fname)
        self.reset()

    @classmethod 
    def from_qdimacs(cls, fname, **kwargs):
        try:
            rc = qdimacs_to_cnf(fname)            
            if rc:
                return cls(rc, **kwargs)
        except:
            print('Error parsing file %s' % fname)

    @property
    def num_vars(self):
        return self.qcnf['maxvar']

    @property
    def num_existential(self):
        a = self.var_types        
        return len(a[a>0])

    @property
    def num_universal(self):
        return self.num_vars - self.num_existential

    @property
    def num_clauses(self):
        k = self.extra_clauses.keys()
        if not k:
            return len(self.qcnf['clauses'])            
        return max(k)+1
    @property
    def max_vars(self):
        try:
            return self._max_vars
        except:
            return self.num_vars
    @property
    def max_clauses(self):
        try:
            return self._max_clauses
        except:
            return self.num_clauses


    # This returns a 0-based numpy array of values per variable up to num_vars. 0 in universal, 1 is existential, 2 is missing

    @property 
    def var_types(self):        
        a = self.qcnf['cvars']
        rc = np.zeros(self.num_vars)+2      # Default var type for "missing" is 2
        for k in a.keys():
            if a[k]['universal']:
                rc[k-1] = 0            
            else:
                rc[k-1] = 1

        return rc.astype(int)


    def get_adj_matrices(self):
        sample = self.qcnf
        if self.sparse:
            return self.get_sparse_adj_matrices()
        else:
            return self.get_dense_adj_matrices()

    def get_sparse_adj_matrices(self):
        sample = self.qcnf
        if self.sp_indices is None:
            clauses = sample['clauses']
            indices = []
            values = []

            for i,c in enumerate(clauses):
                if i in self.removed_old_clauses:
                    continue
                for v in c:
                    val = np.sign(v)
                    v = abs(v)-1            # We read directly from file, which is 1-based, this makes it into 0-based
                    indices.append(np.array([i,v]))
                    values.append(val)

            self.sp_indices = np.vstack(indices)
            self.sp_vals = np.stack(values)
        if not self.extra_clauses:
            return self.sp_indices, self.sp_vals
        indices = []
        values = []            
        for i, c in self.extra_clauses.items():            
            for v in c:
                val = np.sign(v)
                v = abs(v)-1            # We read directly from file, which is 1-based, this makes it into 0-based
                indices.append(np.array([i,v]))
                values.append(val)        
        # Tracer()()
        return np.concatenate([self.sp_indices,np.asarray(indices)]), np.concatenate([self.sp_vals, np.asarray(values)])
        
    def get_dense_adj_matrices(self):
        sample = self.qcnf
        clauses = sample['clauses']          
        new_all_clauses = []        

        rc = np.zeros([self.max_clauses, self.max_vars])
        for i in range(self.num_clauses):
            for j in clauses[i]:                
                t = abs(j)-1
                rc[i][t]=sign(j)
        return rc
    
    def get_base_embeddings(self):
        embs = np.zeros([self.num_vars,GROUND_DIM])
        for i in (IDX_VAR_UNIVERSAL, IDX_VAR_EXISTENTIAL):
            embs[:,i][np.where(self.var_types==i)]=1
        return embs

    def get_clabels(self):
        rc = np.ones(self.num_clauses)
        rc[:len(self.qcnf['clauses'])]=0
        return rc

    def add_clause(self,clause, clause_id):
        assert(clause_id not in self.extra_clauses.keys())
        self.extra_clauses[clause_id]=clause

    def remove_clause(self, clause_id):
        if not (clause_id in self.extra_clauses.keys()):            
            self.removed_old_clauses.append(clause_id)
        else:
            del self.extra_clauses[clause_id]

    @property
    def label(self):
        return 0 if 'UNSAT' in self.qcnf['fname'].upper() else 1

    def as_tensor_dict(self):
        rc = {'sparse': torch.Tensor([int(self.sparse)])}
        
        if self.sparse:
            rc_i, rc_v = self.get_sparse_adj_matrices()
            sp_ind_pos = torch.from_numpy(rc_i[np.where(rc_v>0)])
            sp_ind_neg = torch.from_numpy(rc_i[np.where(rc_v<0)])
            sp_val_pos = torch.ones(len(sp_ind_pos))
            sp_val_neg = torch.ones(len(sp_ind_neg))

            rc['sp_v2c_pos'] = torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([self.max_clauses,self.max_vars]))
            rc['sp_v2c_neg'] = torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([self.max_clauses,self.max_vars]))
        
        rc['v2c'] = torch.from_numpy(self.get_dense_adj_matrices())
        return rc
    
    def as_np_dict(self):
        rc = {}
                
        rc_i, rc_v = self.get_sparse_adj_matrices()
        
        # rc['sp_ind_pos'] = rc_i[np.where(rc_v>0)]
        # rc['sp_ind_neg'] = rc_i[np.where(rc_v<0)]
        rc['sp_indices'] = rc_i
        rc['sp_vals'] = rc_v
        rc['var_types'] = self.var_types
        rc['num_vars'] = self.num_vars
        rc['num_clauses'] = self.num_clauses
        rc['ground'] = self.get_base_embeddings()
        rc['clabels'] = self.get_clabels()
        rc['label'] = self.label
                
        return rc


f2qbf = lambda x: QbfBase.from_qdimacs(x)

class QbfDataset(Dataset):
    def __init__(self, fnames=None, max_variables=MAX_VARIABLES, max_clauses=MAX_CLAUSES):
        self.samples = ([], [])     # UNSAT, SAT
        self.max_vars = max_variables
        self.max_clauses = max_clauses        
        if fnames:
            if type(fnames) is list:
                self.load_files(fnames)
            else:
                self.load_files([fnames])

    def load_dir(self, directory):
        self.load_files([join(directory, f) for f in listdir(directory)])

    def load_files(self, files):
        only_files = [x for x in files if os.path.isfile(x)]
        only_dirs = [x for x in files if os.path.isdir(x)]
        for x in only_dirs:
            self.load_dir(x)
        rc = map(f2qbf,only_files)
        rc = [x for x in rc if x and x.num_vars <= self.max_vars and x.num_clauses < self.max_clauses\
                                                             and x.num_clauses > 0 and x.num_vars > 0]
        for x in rc:            
            self.samples[x.label].append(x)
        try:
            del self.__weights_vector
        except:
            pass
        return len(rc)

    @property
    def num_sat(self):
        return len(self.samples[1])

    @property
    def num_unsat(self):
        return len(self.samples[0])

    @property
    def weights_vector(self):
        try:
            return self.__weights_vector
        except:
            pass

        rc = []
        a =[[1/x]*x for x in [self.num_unsat, self.num_sat]]
        a = np.concatenate(a) / 2
        self.__weights_vector = a
        return a

    def load_file(self,fname):
        if os.path.isdir(fname):
            self.load_dir(fname)
        else:
            self.load_files([fname])

    def get_files_list(self):
        return [x.qcnf['fname'] for x in self.samples[0]] + [x.qcnf['fname'] for x in self.samples[1]]        

    def __len__(self):
        return self.num_unsat + self.num_sat

    def __getitem__(self, idx):
        if idx < self.num_unsat:
            return self.samples[0][idx].as_np_dict()
        else:
            return self.samples[1][idx-self.num_unsat].as_np_dict()


# Obselete, used only in qbf_train.py

# def qbf_collate(batch):
#     rc = {}

#     # Get max var/clauses for this batch    
#     v_size = max([b['num_vars'] for b in batch])
#     c_size = max([b['num_clauses'] for b in batch])

#     # adjacency matrix indices in one huge matrix
#     rc_i = np.concatenate([b['sp_indices'] + np.asarray([i*c_size,i*v_size]) for i,b in enumerate(batch)], 0)
#     rc_v = np.concatenate([b['sp_vals'] for b in batch], 0)

#     # make var_types into ground embeddings
#     all_embs = []
#     for i,b in enumerate(batch):
#         embs = b['ground']
#         l = len(embs)
#         embs = np.concatenate([embs,np.zeros([v_size-l,GROUND_DIM])])
#         all_embs.append(embs)    

#     # break into pos/neg
#     sp_ind_pos = torch.from_numpy(rc_i[np.where(rc_v>0)])
#     sp_ind_neg = torch.from_numpy(rc_i[np.where(rc_v<0)])
#     sp_val_pos = torch.ones(len(sp_ind_pos))
#     sp_val_neg = torch.ones(len(sp_ind_neg))


#     rc['sp_v2c_pos'] = torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([c_size*len(batch),v_size*len(batch)]))
#     rc['sp_v2c_neg'] = torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([c_size*len(batch),v_size*len(batch)]))
#     rc['ground'] = torch.from_numpy(np.stack(all_embs))
#     rc['label'] = torch.Tensor([x['label'] for x in batch]).long()

#     return rc
            
class AagBase(object):    
    """
    Wrapper object for an `aag` circuit (read from a `.qaiger` file).
    """
    def __init__(self, aag = None, **kwargs):
        self.sparse = kwargs['sparse'] if 'sparse' in kwargs else True
        self.aag = aag
        self.sp_indices = None
        self.sp_vals = None
        self.extra_clauses = {}
        self.removed_old_clauses = []
        # self.get_base_embeddings = lru_cache(max_size=16)(self.get_base_embeddings)
        if 'max_variables' in kwargs:
            self._max_vars = kwargs['max_variables']
        if 'max_clauses' in kwargs:
            self._max_clauses = kwargs['max_clauses']

    def reset(self):
        self.sp_indices = None
        self.sp_vals = None
        self.extra_clauses = {}
        
    def load_qaiger(self, filename):
        self.aag = read_qaiger(filename)
        self.reset()

    def get_adj_matrices(self):
        sample = self.aag
        if self.sparse:
            return self.get_sparse_adj_matrices(sample)
        else:
            return self.get_dense_adj_matrices(sample)

    def get_sparse_adj_matrices(self):
        sample = self.aag
        indices = []
        values = []
        
        # should i include the inputs as "clauses", or just the and gates ???
        
        for i, ag in enumerate(sample['and_gates']):
            for l in ag:
                val = 1 if l % 2 == 0 else -1
                indices.append( [i, int(l/2)] )
                values.append(val)
        
        return [indices, np.array(values)]
        
        
#    def get_dense_adj_matrices(self):
#        sample = self.aag
#        clauses = sample['clauses']          
#        new_all_clauses = []        
#
#        rc = np.zeros([self.max_clauses, self.max_vars])
#        for i in range(self.num_clauses):
#            for j in clauses[i]:                
#                t = abs(j)-1
#                rc[i][t]=sign(j)
#        return rc
    

def read_qaiger(filename):
    """
    read from a `.qaiger` file (which contains an `aag` circuit).
    returns a dictionary..
    """
    maxvar = 0
    num_inputs = 0
    num_latches = 0
    num_outputs = 0
    num_and_gates = 0
    inputs = []
    latches = []
    outputs = []
    and_gates = []
    input_symbols = []
    output_symbols = []
    avars = {}

    with open(filename, 'r') as f:
        while True:
            a = f.readline()        # header
            if a[0:3] == 'aag':
                break
        line = a.split(' ')         
        # read the first line, like "aag 256 32 0 1 224"
        maxvar = int(line[1])
        num_inputs = int(line[2])
        num_latches = int(line[3])
        num_outputs = int(line[4])
        num_and_gates = int(line[5])
        
        # initialize avars
        for v in range(maxvar):
            avars[v + 1] =  {'universal': 'Not yet implemented', 'and_gates': []}
        
        # read inputs
        k = num_inputs
        while k > 0 and a:
            a = f.readline()
            line = a.split()
            inputs.append(int(line[0]))
            k -= 1
        
        # ignore latches, for now
        
        # read outputs
        k = num_outputs
        while k > 0 and a:
            a = f.readline()
            line = a.split()
            outputs.append(int(line[0]))
            k -= 1
        
        # read and gates
        k = num_and_gates
        while k > 0 and a:
            a = f.readline()
            line = a.split()
            and_gate = [int(line[0]), int(line[1]), int(line[2])]
            and_gates.append(and_gate)
            k -= 1
            
            # update avars
            for l in and_gate:
                v = int(l/2) # the variable v corresponding to the literal l
                avars[v]['and_gates'].append(and_gate)
            
        # read input symbols
        k = num_inputs
        while k > 0 and a:
            a = f.readline()
            line = a.split()
            input_symbols.append( ' '.join(line[1:]) )
            k -= 1
            
        # read output symbols
        k = num_outputs
        while k > 0 and a:
            a = f.readline()
            line = a.split()
            output_symbols.append( ' '.join(line[1:]) )
            k -= 1
        
    return {'maxvar': maxvar,
            'num_inputs': num_inputs,
            'num_latches': num_latches,
            'num_outputs': num_outputs,
            'num_and_gates': num_and_gates,
            'inputs': inputs,
            'latches': latches,
            'outputs': outputs,
            'and_gates': and_gates,
            'input_symbols': input_symbols,
            'output_symbols': output_symbols,
            #'avars': avars,
            'fname': filename
            }
