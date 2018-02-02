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
import pdb

_use_shared_memory = False

class QbfBase(object):    
    def __init__(self, qcnf = None, **kwargs):
        self.sparse = kwargs['sparse'] if 'sparse' in kwargs else False
        self.qcnf = qcnf

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
        return self.num_vars
    @property
    def max_clauses(self):
        return self.num_clauses

    def get_adj_matrices(self,sample):
        if self.sparse:
            return self.get_sparse_adj_matrices(sample)
        else:
            return self.get_dense_adj_matrices(sample)

    def get_sparse_adj_matrices(self,sample):
        clauses = sample['clauses']        
        cvars = sample['cvars']                            
        indices = []
        values = []        

        for i,c in enumerate(clauses):
            for v in c:
                val = 1 if v>0 else -1
                v = abs(v)-1
                indices.append(np.array([i,v]))
                values.append(val)

        sp_indices = np.vstack(indices)
        sp_vals = np.stack(values)
        return sp_indices, sp_vals
        
    def get_dense_adj_matrices(self,sample):
        clauses = sample['clauses']        
        cvars = sample['cvars']                            
        new_all_clauses = []        

        for i in range(self.max_clauses):
            new_clause = np.zeros(self.max_vars)
            if i<len(clauses):
                x = clauses[i]
                for j in range(self.max_vars):
                    t = j+1
                    if t in x:
                        new_clause[j]=1
                    elif -t in x:                        
                        new_clause[j]=-1
                new_all_clauses.append(new_clause)
            else:                
                new_all_clauses.append(new_clause)
        if len(new_all_clauses) != self.max_clauses:
            import ipdb; ipdb.set_trace()
        
        v2c = np.stack(new_all_clauses)        
        return v2c


    def as_tensor_dict(self):
        rc = {'sparse': torch.Tensor([int(self.sparse)])}
        
        if self.sparse:
            rc_i, rc_v = self.get_sparse_adj_matrices(self.qcnf)
            sp_ind_pos = torch.from_numpy(rc_i[np.where(rc_v>0)])
            sp_ind_neg = torch.from_numpy(rc_i[np.where(rc_v<0)])
            sp_val_pos = torch.ones(len(sp_ind_pos))
            sp_val_neg = torch.ones(len(sp_ind_neg))

            rc['sp_v2c_pos'] = torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([self.num_clauses,self.num_vars]))
            rc['sp_v2c_neg'] = torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([self.num_clauses,self.num_vars]))
        else:
            rc['v2c'] = torch.from_numpy(self.get_dense_adj_matrices(self.qcnf))

        return rc
