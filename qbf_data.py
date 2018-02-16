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

MAX_VARIABLES = 50
MAX_CLAUSES = 500
GROUND_DIM = 5
IDX_VAR_UNIVERSAL = 0
IDX_VAR_EXISTENTIAL = 1
IDX_VAR_MISSING = 2
IDX_VAR_DETERMINIZED = 3
IDX_VAR_ACTIVITY = 4

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
        return rc.astype(int)


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
    
    def as_np_dict(self):
        rc = {}
                
        rc_i, rc_v = self.get_sparse_adj_matrices(self.qcnf)
        
        # rc['sp_ind_pos'] = rc_i[np.where(rc_v>0)]
        # rc['sp_ind_neg'] = rc_i[np.where(rc_v<0)]
        rc['sp_indices'] = rc_i
        rc['sp_vals'] = rc_v
        rc['var_types'] = self.var_types
        rc['num_vars'] = self.num_vars
        rc['num_clauses'] = self.num_clauses
                
        return rc


f2qbf = lambda x: QbfBase.from_qdimacs(x)

class QbfDataset(Dataset):
    def __init__(self,dirname=None, fnames=None, max_variables=MAX_VARIABLES, max_clauses=MAX_CLAUSES):
        self.samples = []
        self.max_vars = max_variables
        self.max_clauses = max_clauses
        if dirname:            
            self.load_dir(dirname)
        elif fnames:
            self.load_files(fnames)

    def load_dir(self, directory):
        self.load_files([join(directory, f) for f in listdir(directory)])

    def load_files(self, files):
        rc = zip(map(f2qbf,files),files)
        rc = [x for x in rc if x[0] and x[0].num_vars <= self.max_vars and x[0].num_clauses < self.max_clauses\
                                                             and x[0].num_clauses > 0 and x[0].num_vars > 0]
        self.samples += rc
        return len(rc)

    def load_file(self,fname):
        self.load_files([fname])

    def get_files_list(self):
        _, rc = zip(*self.samples)
        return rc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        print('getting idx %d' % idx)
        return self.samples[idx][0].as_np_dict()

def qbf_collate(batch):
    rc = {}

    # Get max var/clauses for this batch    
    v_size = max([b['num_vars'] for b in batch])
    c_size = max([b['num_clauses'] for b in batch])

    # adjacency matrix indices in one huge matrix
    rc_i = np.concatenate([b['sp_indices'] + np.asarray([i*c_size,i*v_size]) for i,b in enumerate(batch)], 0)
    rc_v = np.concatenate([b['sp_vals'] for b in batch], 0)

    # make var_types into ground embeddings
    embs = np.zeros([len(batch),v_size,GROUND_DIM])
    for i,b in enumerate(batch):
        for j, val in enumerate(b['var_types']):
            embs[i][j][val] = True
        embs[i,b['num_vars']:,IDX_VAR_MISSING] = True

    # break into pos/neg
    sp_ind_pos = torch.from_numpy(rc_i[np.where(rc_v>0)])
    sp_ind_neg = torch.from_numpy(rc_i[np.where(rc_v<0)])
    sp_val_pos = torch.ones(len(sp_ind_pos))
    sp_val_neg = torch.ones(len(sp_ind_neg))


    rc['sp_v2c_pos'] = torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([c_size*len(batch),v_size*len(batch)]))
    rc['sp_v2c_neg'] = torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([c_size*len(batch),v_size*len(batch)]))
    rc['ground'] = torch.from_numpy(embs)
    return rc

