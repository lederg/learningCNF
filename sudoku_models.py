import torch
import numpy as np
import ipdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as nn_init

from settings import *

def patch_grid(grid, indices, vals):
	np.put(grid,np.ravel_multi_index(indices.T,grid.shape),vals)


# indices is array of dimension 3 indices, grid is dimension 4, where 4th dimension is embedding
# return indices.shape[0] x grid.shape[-1]
def get_from_grid(grid, indices):
	embdim = grid.shape[-1]
	ind = np.ravel_multi_index(indices.T,grid.shape[:-1])
	a = grid.reshape(-1, embdim)
	return a[ind,:]

class SudokuModel1(nn.Module):
  def __init__(self, size=9):
    super(SudokuModel1, self).__init__()
    self.settings = CnfSettings()
    self.decoded_dim = self.settings['sharp_decoded_emb_dim']
    self.size = size
    self.layer1 = nn.Sequential(
    	nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
    	nn.ReLU(), 
    	nn.Conv3d(16, 16, kernel_size=5, stride=1, padding=2),
    	nn.ReLU(), 
    	nn.Conv3d(16, self.decoded_dim, kernel_size=3, stride=1, padding=1),
    	nn.ReLU()) 



  def forward(self, input_tensor):
  	inp = input_tensor.unsqueeze(0).unsqueeze(0).detach()			# Add batch and channel dimensions  	
  	out = self.layer1(inp).squeeze(0).transpose(0,3)
  	return out