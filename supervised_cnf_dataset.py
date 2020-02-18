import os
import sys
import ipdb
import torch
import pickle
import dgl
import numpy as np
# from lru import LRU
from torchvision import transforms
from torch.utils.data import Dataset
from utils_dir.utils import load_dir, load_files
from sat_code.supervised import get_graph

CACHE_SIZE = 200


def sat_collate_fn(batch):
  rc = {}
  rc['gss'] = torch.cat([x['gss'] for x in batch],dim=0)
  rc['graph'] = dgl.batch_hetero([x['graph'] for x in batch])
  rc['vnum'] = [x['vnum'] for x in batch]
  rc['cnum'] = [x['cnum'] for x in batch]

  labels = (rc['graph'].nodes['clause'].data['clause_targets'][:,1] == 0).long()
  learnt = (rc['graph'].nodes['clause'].data['clause_labels'][:,-1]).int()
  pred_idx = torch.where(rc['graph'].nodes['clause'].data['predicted_clauses'])[0]
  # print("collate says, pred_idx (out of {}) is:".format(labels.size(0)))
  # print(pred_idx)

  labels_learnt = labels[pred_idx]

  return rc, labels_learnt

def load_formula(fname):
  with open(fname,'rb') as f:
    formula = pickle.load(f)
  g_dgl = get_graph(formula['adjacency'], torch.Tensor(formula['clause_feat_and_labels']), torch.Tensor(formula['lit_feat']))
  rc = {}
  if formula['lit_feat'].max() == np.inf:
    ipdb.set_trace()
  rc['graph'] = g_dgl
  rc['vnum'] = g_dgl.number_of_nodes('literal')
  rc['cnum'] = g_dgl.number_of_nodes('clause')
  rc['gss'] = torch.Tensor(formula['gss']).expand(rc['cnum'], len(formula['gss']))
  return rc

class SampleLearntClauses(object):
  def __init__(self,num):
    self.num = num
  def __call__(self, sample):
    G = sample['graph']
    labels = (G.nodes['clause'].data['clause_targets'][:,1] == 0).long()
    learnt = (G.nodes['clause'].data['clause_labels'][:,-1]).int()
    learnt_idx = torch.where(learnt)[0]
    labels_learnt = labels[learnt_idx]
    labels_pos = torch.where(labels_learnt)[0]
    labels_neg = torch.where(labels_learnt==0)[0]
    pos_idx = learnt_idx[labels_pos[torch.torch.randperm(labels_pos.size(0))[:self.num]]]
    neg_idx = learnt_idx[labels_neg[torch.torch.randperm(labels_neg.size(0))[:self.num]]]
    predicted_idx = torch.cat([pos_idx,neg_idx],dim=0)
    predicted_arr = torch.zeros(labels.size(0))
    predicted_arr[predicted_idx] = 1
    G.nodes['clause'].data['predicted_clauses'] = predicted_arr    
    return sample

class CapActivity(object):
  def __call__(self, sample):
    G = sample['graph']
    G.nodes['literal'].data['lit_labels'][:,3].tanh_()
    G.nodes['clause'].data['clause_labels'][:,3].tanh_()
    return sample
        
class ZeroClauseIndex(object):
  def __init__(self, index):
    self.index = index
  def __call__(self, sample):
    G = sample['graph']    
    G.nodes['clause'].data['clause_labels'][:,self.index] = 0
    return sample

class ZeroLiteralIndex(object):
  def __init__(self, index):
    self.index = index
  def __call__(self, sample):
    G = sample['graph']    
    G.nodes['literal'].data['lit_labels'][:,self.index] = 0
    return sample


class CnfGNNDataset(Dataset):
  def __init__(self, fname, transform=lambda x: x, cap_size=sys.maxsize):
    self.items = load_files(fname)    
    # self.cache = LRU(cache_size)
    self.transform = transform

  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    fname = self.items[idx]
    return self.transform(load_formula(fname))
    # if fname not in self.cache:     
    #   self.cache[fname] = self.transform(load_formula(fname))
    # return self.cache[fname]
