import ipdb
import time
import logging

from settings import *
from utils import Singleton


def mhp1_2_mhp2(policy, saved_dict):
	sd = policy.state_dict()
	for k in sd:
		if k == 'policy_layers.linear_2.weight':
			sd[k][0,:]=saved_dict[k][0,:]
			sd[k][1,:]=0
			sd[k][2,:]=saved_dict[k][1,:]
		elif k == 'policy_layers.linear_2.bias':
			sd[k][0]=saved_dict[k][0]
			sd[k][1]=0
			sd[k][2]=saved_dict[k][1]
			pass
		else:
			sd[k] = saved_dict[k]
	policy.load_state_dict(sd)
