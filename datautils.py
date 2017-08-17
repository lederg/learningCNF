from eqnet_parser import *
import numpy as np
from functools import partial
from torch.utils.data import Dataset
import os

class CnfDataset(Dataset):    
    def __init__(self, json_file, threshold=10):
        self.CLASS_THRESHOLD = threshold

        self.eq_classes = self.filter_classes(to_cnf(load_bool_data(json_file)))
        self.labels = list(self.eq_classes.keys())
        self.samples = list(self.eq_classes.values())        
        self.class_size = [len(x) for x in self.samples]
        self.class_cumsize = np.cumsum(self.class_size)        
    def __len__(self):
        return sum(self.class_size)

    def __getitem__(self, idx):        
        i = np.where(self.class_cumsize > idx)[0][0]            # This is the equivalence class
        j = idx if i==0 else idx-self.class_cumsize[i-1]        # index inside equivalence class
        sample = self.samples[i][j]        
        sample, topvar = self.transform_sample(sample)

        return {'sample': sample, 'label': i, 'topvar': topvar}

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



    def transform_sample(self,sample):
        clauses = sample['clauses_per_variable']
        auxvars = sample['auxvars']        
        origvars = list(sample['origvars'].values())
        num_total_vars = len(origvars) + len(auxvars)
        rc = []

        def convert_var(v):
            j = abs(v)
            if j<=self.ground_vars: return v
            if j in auxvars:
                newvar = self.ground_vars+auxvars.index(j)+1
                return newvar if v>0 else -newvar 
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

        return rc, convert_var(sample['topvar'])



    def filter_classes(self,classes):
        a = {k: v for k,v in classes.items() if len(v) > self.CLASS_THRESHOLD}
        m = np.mean([len(x) for x in a.values()])
        rc = {k: v for k,v in a.items() if len(v) < 3*m}
        rc1 = {}
        for k,v in rc.items():
            v1 = [x for x in v if x['clauses_per_variable']]
            if len(v1) < len(v):
                print('removed empty %d formulas from key %s' % (len(v)-len(v1),k))
            rc[k] = v1
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
    def num_classes(self):
        return len(self.labels)