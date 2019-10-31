from cnf_parser import *
from aag_parser import read_qaiger
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
import dgl

_use_shared_memory = False

MAX_VARIABLES = 1000000
MAX_CLAUSES = 5000000
# MAX_VARIABLES = 10000
# MAX_CLAUSES = 50000
GROUND_DIM = 8          # config.ground_dim duplicates this. # changed for DGL branch 
IDX_VAR_UNIVERSAL = 0
IDX_VAR_EXISTENTIAL = 1
IDX_VAR_INPUT_OUTPUT = 8 # Last Column
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

##############################################################################
            
class CombinedGraph1Base(object):    
    """
    Wrapper object for paired files ('a.qaiger', 'a.qdimacs) 
    Holds the combined AAG-CNF graph (our first implementation, Oct 2019)
    Uses the DGL package to store this heterogeneous graph
        3 types of nodes: AAG literals, QCNF literals, QCNF clauses
        4 types of edges: 
            AAG literal -> AAG literal forward edges 
            AAG literal -> AAG literal backward edges 
            QCNF literal -> QCNF clause forward edges 
            QCNF clause -> QCNF literal backward edges
        9-dimensional node features for AAG literals and QCNF literals
    """
    def __init__(self, qcnf = None, qcnf_base = None, aag = None, dgl_ = None, **kwargs):
        self.qcnf = qcnf
        self.qcnf_base = qcnf_base
        self.aag = aag
        self.dgl_ = dgl_
        self.GROUNDDIM = 9
        self.IDX_VAR_UNIVERSAL = 0
        self.IDX_VAR_EXISTENTIAL = 1
        self.IDX_VAR_INPUT_OUTPUT = 8 # Last Column
        
    def load_paired_files(self, aag_fname = None, qcnf_fname = None):
        assert aag_fname.endswith('.qaiger') and qcnf_fname.endswith('.qdimacs')
        self.aag_og = read_qaiger(aag_fname)        # for testing that file reading works
        self.qcnf_og = qdimacs_to_cnf(qcnf_fname)   # for testing that file reading works
        self.aag = self.fix_aag_numbering(read_qaiger(aag_fname))
        self.qcnf = self.fix_qcnf_numbering(qdimacs_to_cnf(qcnf_fname))
        self.dgl_ = self.initialize_DGL_graph()
        
    def fix_aag_numbering(self, aag):
        """
        .qaiger file read in with variables 1,2,...,n and literals 2,3,4,5,...,2n,2n+1.
        Change  literals 2,  3, 4,  5, ..., 2n,  2n+1
        to      literals 0,  1, 2,  3, ..., 2n-2,2n-1 
        in order to make the literals 0-based.
        """
        for i, inp in enumerate(aag['inputs']):
            aag['inputs'][i] = int(inp - 2)            
        for i, out in enumerate(aag['outputs']):
            aag['outputs'][i] = int(out - 2)
        for ag in aag['and_gates']:
            for i, e in enumerate(ag):
                ag[i] = int(e - 2)
        return aag
    
    def convert_qdimacs_lit(self, lit):
        L = 2 * abs(lit)
        L = L + 1 if (lit < 0) else L
        return int(L - 2)
    
    def fix_qcnf_numbering(self, qcnf):
        """
        .qdimacs file read in with variables 1,2,...,n and literals 1,-1,2,-2,...,n,-n.
        Change  literals 1, -1, 2, -2, ..., n,   -n
        to      literals 2,  3, 4,  5, ..., 2n,  2n+1
        to      literals 0,  1, 2,  3, ..., 2n-2,2n-1 
        in order to make the literals 0-based.
        """
        # change literal numbers: in 'clauses'
        for cl in qcnf ['clauses']:
            for i, lit in enumerate(cl):
                cl[i] = self.convert_qdimacs_lit(lit)
                
        # change variable numbers: keys of 'cvars'
        cvars = {}
        for key in qcnf['cvars']:
            cvars[int(key - 1)] = qcnf['cvars'][key]
        qcnf['cvars'] = cvars
        
        return qcnf
    
    def initialize_DGL_graph(self):
        """
        Create the combined AAG-QCNF graph.
        """
        # create aag edges, which remain fixed
        aag_forward_edges, aag_backward_edges = [], []
        for ag in self.aag['and_gates']:
            x, y, z = ag[0], ag[1], ag[2]
            aag_forward_edges.append( (x,y) )
            aag_forward_edges.append( (x,z) )
            aag_backward_edges.append( (y,x) )
            aag_backward_edges.append( (z,x) )
                        
        # create qcnf edges, which will need to be updated from CADET obs
        qcnf_forward_edges, qcnf_backward_edges = [], []
        for cl_num, clause in enumerate(self.qcnf['clauses']):
            for lit in clause:
                qcnf_forward_edges.append( (lit,cl_num) )
                qcnf_backward_edges.append( (cl_num,lit) )
                            
        G = dgl.heterograph(
            {('aag_lit', 'aag_forward', 'aag_lit') : aag_forward_edges,
             ('aag_lit', 'aag_backward', 'aag_lit') : aag_backward_edges,
             ('qcnf_lit', 'qcnf_forward', 'qcnf_clause') : qcnf_forward_edges,
             ('qcnf_clause', 'qcnf_backward', 'qcnf_lit') : qcnf_backward_edges},
             
            {'aag_lit': 2 * self.aag['maxvar'],
             'qcnf_lit' : 2 * self.qcnf['maxvar'],
             'qcnf_clause': self.qcnf['num_clauses']}
        )
        
        G.nodes['aag_lit'].data['aag_lit_embs'] = self.initial_aag_features()
        G.nodes['qcnf_lit'].data['qcnf_lit_embs'] = self.initial_qcnf_features()
        return G
    
    
    def initial_qcnf_features(self):
        """
        9 dimensional features for 'qcnf_lit' nodes. Shape (num_nodes, num_features=9).
        1st column: 1 iff lit is universal
        2nd column: 1 iff lit is existential
        9th column: 1 iff lit is an input or output [NOT YET IMPLEMENTED]
        """
        universal_lits, existential_lits = [], []
        for v in self.qcnf['cvars'].keys():
            if self.qcnf['cvars'][v]['universal']:
                universal_lits.append(2*v)
                universal_lits.append(2*v+1)
            else:
                existential_lits.append(2*v)
                existential_lits.append(2*v+1)
        embs = np.zeros([2 * self.qcnf['maxvar'], self.GROUNDDIM]) 
        embs[:, self.IDX_VAR_UNIVERSAL][universal_lits] = 1
        embs[:, self.IDX_VAR_EXISTENTIAL][existential_lits] = 1
        return embs
        
    
    def initial_aag_features(self):
        """
        9 dimensional features for 'qcnf_lit' nodes. Shape (num_nodes, num_features=9).
        1st column: 1 iff lit is universal [NOT YET IMPLEMENTED]
        2nd column: 1 iff lit is existential [NOT YET IMPLEMENTED]
        9th column: 1 iff (lit or (not lit)) is an input or output 
        """
        embs = torch.zeros([2 * self.aag['maxvar'], self.GROUNDDIM]) 
        embs[:, self.IDX_VAR_INPUT_OUTPUT][self.aag['inputs']] = 1
        embs[:, self.IDX_VAR_INPUT_OUTPUT][self.aag['outputs']] = 1
        try:
            embs[:, self.IDX_VAR_INPUT_OUTPUT][np.array(self.aag['inputs'])+1] = 1  # DELETE THIS?
            embs[:, self.IDX_VAR_INPUT_OUTPUT][np.array(self.aag['outputs'])+1] = 1  # DELETE THIS?
        except:
            pass
        return embs
    
##############################################################################
##### TESTING    
        
#a = CombinedGraph1Base()
#a.load_paired_files(aag_fname = './data/words_test_ryan/words_ry_SAT.qaiger', qcnf_fname = './data/words_test_ryan/words_ry_SAT.qaiger.qdimacs')

