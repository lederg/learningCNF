from eqnet_parser import *
import numpy as np
from functools import partial
from torch.utils.data import Dataset
from enum import Enum
import os
import random
import pdb

class DataMode(Enum):
    NORMAL = 1
    SAT=2
    TRENERY=3
    TF=4



class CnfDataset(Dataset):    
    def __init__(self, json_file, threshold=10, ref_dataset=None, mode: DataMode=DataMode.NORMAL):
        self.CLASS_THRESHOLD = threshold

        if mode == DataMode.TRENERY:
            self.eq_classes = self.trenery_filter_classes(to_cnf(load_bool_data(json_file)))            
        elif mode == DataMode.SAT:
            self.eq_classes = self.sat_filter_classes(to_cnf(load_bool_data(json_file)))            
        elif mode == DataMode.TF:
            self.eq_classes = self.tf_filter_classes(to_cnf(load_bool_data(json_file)))            
        elif mode == DataMode.NORMAL:
            if not ref_dataset:
                self.eq_classes = self.filter_classes(to_cnf(load_bool_data(json_file)))
            else:
                self.eq_classes = self.filter_classes_by_ref(to_cnf(load_bool_data(json_file)),ref_dataset)
        # self.eq_classes = self.dummy_filter(to_cnf(load_bool_data(json_file)))
        self.labels = list(self.eq_classes.keys())
        self.samples = list(self.eq_classes.values())        
        self.class_size = [len(x) for x in self.samples]
        self.class_cumsize = np.cumsum(self.class_size) 
        self.cache = np.empty(sum(self.class_size),dtype=object)
    def __len__(self):
        return sum(self.class_size)

    def __getitem__(self, idx):
        if self.cache[idx]:
            return self.cache[idx]
        i = np.where(self.class_cumsize > idx)[0][0]            # This is the equivalence class
        j = idx if i==0 else idx-self.class_cumsize[i-1]        # index inside equivalence class
        orig_sample = self.samples[i][j]        
        variables, clauses, topvar = self.transform_sample(orig_sample)
        self.cache[idx] = {'variables': variables, 'clauses': clauses, 'label': i, 'topvar': topvar, 'idx_in_dataset': idx}
        # self.cache[idx] = {'topvar': topvar}
        return self.cache[idx]
        

    @property
    def weights_vector(self):
        try:
            return self.__weights_vector
        except:
            pass

        rc = []
        a =[[1/x]*x for x in self.class_size]
        a = np.concatenate(a) / len(self.class_size)     # a now holds the relative size of classes
        self.__weights_vector = a
        return a


    def get_class_indices(self,c):
        if c==0:
            return range(self.class_cumsize[0])
        else:
            return range(self.class_cumsize[c-1],self.class_cumsize[c])


    def transform_sample(self,sample):
        clauses = sample['clauses_per_variable']
        auxvars = sample['auxvars']
        topvar = sample['topvar']
        
        origvars = list(sample['origvars'].values())
        num_total_vars = len(origvars) + len(auxvars)
        rc = []
        rc1 = []

        def convert_var(v):
            j = abs(v)
            if j<=self.ground_vars: return v            
            if j in auxvars:
                newvar = self.ground_vars+auxvars.index(j)+1
                rc = newvar if v>0 else -newvar 
                if topvar < 0 and abs(v) == abs(topvar):                  # We invert the topvar variable if its nega
                    return -rc
                else:
                    return rc
                    
            else:
                print('What the heck?')
                import ipdb; ipdb.set_trace()

        # First append ground vars or empty clauses. 
        for i in range(1,self.ground_vars+1):     # we are appending anyway, so who cares about index        
            if i in origvars:            
                try:
                    rc.append([list(map(convert_var,x)) for x in clauses[i]])
                except:
                    import ipdb; ipdb.set_trace()
            else:
                rc.append([])

        # No empty clauses for auxvars
        for i in auxvars:
            rc.append([list(map(convert_var,x)) for x in clauses[i]])

        all_clauses = [list(map(convert_var,x)) for x in sample['clauses']]
        for i,v in enumerate(rc):
            cl = []         # new list of indices for clauses
            for c in v:     # for each clause, add either its index or negative index
                idx = all_clauses.index(c) + 1      # clauses (and variables) are 1 based
                if not i+1 in c:
                    idx = -idx
                cl.append(idx)
            rc1.append(cl)

        new_all_clauses = []
        new_all_variables = []
        for i in range(self.max_clauses):
            new_clause = np.zeros(self.max_variables)
            if i<len(all_clauses):
                x = all_clauses[i]
                for j in range(self.max_variables):
                    t = j+1
                    if t in x:
                        new_clause[j]=1
                    elif -t in x:
                        new_clause[j]=2
                new_all_clauses.append(new_clause)
            else:                
                new_all_clauses.append(new_clause)
        if len(new_all_clauses) != self.max_clauses:
            import ipdb; ipdb.set_trace()

        for i in range(self.max_variables):
            new_var = np.zeros(self.max_clauses)
            if i<len(rc1):
                x = rc1[i]
                for j in range(self.max_clauses):
                    t = j+1
                    if t in x:
                        new_var[j]=1
                    elif -t in x:
                        new_var[j]=2
                new_all_variables.append(new_var)
            else:
                new_all_variables.append(new_var)

        return np.stack(new_all_variables), np.stack(new_all_clauses), convert_var(sample['topvar'])


    def dummy_filter(self, classes):
        return {'b': classes['b'], 'a': classes['a']}

    def filter_classes_by_ref(self,classes, ref_dataset):
        return {k: v for k,v in classes.items() if k in ref_dataset.labels}

    def filter_classes(self,classes):
        a = {k: v for k,v in classes.items() if len(v) > self.CLASS_THRESHOLD}
        m = np.mean([len(x) for x in a.values()])
        rc = a
        # rc = {k: v for k,v in a.items() if len(v) < 3*m}
        rc1 = {}
        for k,v in rc.items():
            v1 = [x for x in v if x['clauses_per_variable']]
            if len(v1) < len(v):
                print('removed empty %d formulas from key %s' % (len(v)-len(v1),k))
            rc[k] = v1
        return rc

    def trenery_filter_classes(self,classes):
        a = {k: v for k,v in classes.items() if len(v) > self.CLASS_THRESHOLD}
        m = np.mean([len(x) for x in a.values()])
        rc = {'Other': []}
        for k,v in a.items():
            v1 = [x for x in v if x['clauses_per_variable']]
            if len(v1) < len(v):
                print('removed empty %d formulas from key %s' % (len(v)-len(v1),k))
            if k in ['True', 'False']:
                rc[k] = v1
            else:
                rc['Other'] += v1
        return rc

    def tf_filter_classes(self,classes):
        a = {k: v for k,v in classes.items() if len(v) > self.CLASS_THRESHOLD}
        m = np.mean([len(x) for x in a.values()])
        rc = {}
        for k,v in a.items():
            v1 = [x for x in v if x['clauses_per_variable']]
            if len(v1) < len(v):
                print('removed empty %d formulas from key %s' % (len(v)-len(v1),k))
            if k in ['True', 'False']:
                rc[k] = v1            
        return rc

    def sat_filter_classes(self,classes):
        a = {k: v for k,v in classes.items() if len(v) > self.CLASS_THRESHOLD}
        m = np.mean([len(x) for x in a.values()])
        rc = {'SAT': []}
        for k,v in a.items():
            v1 = [x for x in v if x['clauses_per_variable']]
            if len(v1) < len(v):
                print('removed empty %d formulas from key %s' % (len(v)-len(v1),k))
            if k in ['False']:
                rc[k] = v1
            else:
                rc['SAT'] += v1
        return rc

    @property
    def ground_vars(self):
        try:
            return self.num_ground_vars
        except:
            pass
        rc = 0
        for x in self.samples:
            for sample in x:
                rc = max(rc,max(sample['origvars'].values()))
        self.num_ground_vars = rc
        return rc
    @property
    def max_clauses(self):
        try:
            return self.num_max_clauses
        except:
            pass
        rc = 0
        for x in self.samples:
            for sample in x:
                rc = max(rc,max([len(x) for x in sample['clauses_per_variable'].values()]))
        self.num_max_clauses = rc
        return rc

    @property
    def max_variables(self):
        try:
            return self.num_max_variables
        except:
            pass
        rc = 0
        for x in self.samples:
            for sample in x:
                rc = max(rc,len(sample['auxvars']))                
        self.num_max_variables = rc + self.ground_vars
        return self.num_max_variables
        
    @property
    def num_classes(self):
        return len(self.labels)



class SiameseDataset(Dataset):    
    def __init__(self, json_file, pairwise_epochs=1, **kwargs):
        self.cnf_ds = CnfDataset(json_file, **kwargs)        
        self.pairs = []
        self.build_pairs(epochs=pairwise_epochs)

    ''' 
        For each epoch we sample N positive and N negative pairs from the internal CnfDataset
    '''

    def build_pairs(self, epochs):
        while epochs > 0:
            epochs -= 1
            for i in range(self.cnf_ds.num_classes):
                x = 0
                y = 0

                # sample two indices of same class
                while x == y:
                    x = random.choice(self.cnf_ds.get_class_indices(i))
                    y = random.choice(self.cnf_ds.get_class_indices(i))
                self.pairs.append({'left': self.cnf_ds[x], 'right': self.cnf_ds[y], 'label': 1})

                # sample two indices of different class
                x = random.choice(self.cnf_ds.get_class_indices(i))
                other_class = i
                while i == other_class:
                    other_class = random.choice(range(self.cnf_ds.num_classes))
                y = random.choice(self.cnf_ds.get_class_indices(other_class))
                self.pairs.append({'left': self.cnf_ds[x], 'right': self.cnf_ds[y], 'label': -1})

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]





