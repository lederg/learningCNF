from cnf_parser import *
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
import ipdb

_use_shared_memory = False

class QbfBase(object):    
    def __init__(self, qcnf = None, **kwargs):
        self.sparse = kwargs['sparse'] if 'sparse' in kwargs else False
        self.qcnf = qcnf
        if 'max_variables' in kwargs:
            self._max_vars = kwargs['max_variables']
        if 'max_clauses' in kwargs:
            self._max_clauses = kwargs['max_clauses']


    def reload_qdimacs(self, fname):
        self.qcnf = qdimacs_to_cnf(fname)

    @classmethod 
    def from_qdimacs(cls, fname, **kwargs):
        return cls(qdimacs_to_cnf(fname), **kwargs)

    @property
    def num_vars(self):
        return self.qcnf['maxvar']
    @property
    def num_clauses(self):
        return self.qcnf['num_clauses']
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
        rc = np.zeros(self.num_vars)+2
        for k in a.keys():
            rc[k-1] = 0 if a[k]['universal'] else 1
        return rc


    def get_adj_matrices(self,sample):
        if self.sparse:
            return self.get_sparse_adj_matrices(sample)
        else:
            return self.get_dense_adj_matrices(sample)

    def get_sparse_adj_matrices(self,sample):
        clauses = sample['clauses']                
        indices = []
        values = []        

        for i,c in enumerate(clauses):
            for v in c:
                val = 1 if v>0 else -1
                v = abs(v)-1            # We read directly from file, which is 1-based, this makes it into 0-based
                indices.append(np.array([i,v]))
                values.append(val)

        sp_indices = np.vstack(indices)
        sp_vals = np.stack(values)
        return sp_indices, sp_vals
        
    def get_dense_adj_matrices(self,sample):
        clauses = sample['clauses']                
        new_all_clauses = []        

        rc = np.zeros([self.max_clauses, self.max_vars])
        for i in range(self.num_clauses):
            for j in clauses[i]:                
                t = (abs(j)-1)*sign(j)
                rc[i][t]=sign(j)
        return rc


    def as_tensor_dict(self):
        rc = {'sparse': torch.Tensor([int(self.sparse)])}
        
        if self.sparse:
            rc_i, rc_v = self.get_sparse_adj_matrices(self.qcnf)
            sp_ind_pos = torch.from_numpy(rc_i[np.where(rc_v>0)])
            sp_ind_neg = torch.from_numpy(rc_i[np.where(rc_v<0)])
            sp_val_pos = torch.ones(len(sp_ind_pos))
            sp_val_neg = torch.ones(len(sp_ind_neg))

            rc['sp_v2c_pos'] = torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([self.max_clauses,self.max_vars]))
            rc['sp_v2c_neg'] = torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([self.max_clauses,self.max_vars]))
        
        rc['v2c'] = torch.from_numpy(self.get_dense_adj_matrices(self.qcnf))
        return rc
