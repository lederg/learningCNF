import os
import random
import time
import noise
import functools
import numpy as np
import aiger as A
import aiger_bv as BV
import aiger_coins as C
import aiger_gridworld as GW
import aiger_ptltl as LTL
import aiger_cnf as ACNF
import aiger.common as cmn
import funcy as fn
import matplotlib.pyplot as plt

from gen_types import FileName
from bidict import bidict
from samplers.sampler_base import SamplerBase
from random import randint, seed
from gen_utils import random_string

def get_mask_test(X, Y):
  def f(xmask, ymask):
    return ((X & xmask) !=0) & ((Y & ymask) != 0)
  return f


def create_sensor(aps):
  sensor = BV.aig2aigbv(A.empty())
  for name, ap in aps.items():
      sensor |= ap.with_output(name).aigbv
  return sensor

def get_noise_grid(base=0, shape=(16,16), scale=100., octaves=6, persistence=0.5, lacunarity=2.0):
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.snoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=shape[0], 
                                        repeaty=shape[1], 
                                        base=base)
    return world

def world_from_noise(noise_grid):
    def map_perlin_to_colors(val):
        if val < -0.27:
            return 3          # 'red'
        elif val < 0.15:
            return 0          # white,
        elif val < 0.2:
            return 1        # yellow
        elif val < 0.38:
            return 4          # blue
        else: 
            return 2
    
    
    world = np.zeros(noise_grid.shape)
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            world[i][j] = map_perlin_to_colors(noise_grid[i][j])
    return world    

def world_to_features(world, start):
  num_feats = np.max(world)
  start_feat = np.zeros_like(world).astype(int)
  start_feat
  np.stack([(world==i).astype(int) for i in range(num_feats)],axis=0)



def spec2monitor(spec):
  monitor = spec.aig | A.sink(['red', 'yellow', 'brown', 'blue'])
  monitor = BV.aig2aigbv(monitor)
  return monitor

def mdp2cnf(circ, horizon, *, fresh=None, truth_strategy='last'):
    if fresh is None:
        max_var = 0

        def fresh(_):
            nonlocal max_var
            max_var += 1
            return max_var
    
    
    imap = circ.imap
    inputs = circ.inputs
    step, old2new_lmap = circ.cutlatches()
    init = dict(old2new_lmap.values())
    init = step.imap.blast(init)
    states = set(init.keys())    
    state_inputs = [A.aig.Input(k) for k in init.keys()]
    clauses, seen_false, gate2lit = [], False, ACNF.cnf.SymbolTable(fresh)
    
    # Set up init clauses
    true_var = fresh(True)    
    clauses.append((true_var,))                
    tf_vars = {True: true_var, False: -true_var}
    for k,v in init.items():
        gate2lit[A.aig.Input(k)] = tf_vars[v]
        
    in2lit = bidict()
    outlits= []
    
    for time in range(horizon):
        # Only remember states.        
        gate2lit = ACNF.cnf.SymbolTable(fresh,fn.project(gate2lit, state_inputs))
        for gate in cmn.eval_order(step.aig):
            if isinstance(gate, A.aig.Inverter):
                gate2lit[gate] = -gate2lit[gate.input]
            elif isinstance(gate, A.aig.AndGate):
                clauses.append((-gate2lit[gate.left], -gate2lit[gate.right],  gate2lit[gate]))  # noqa
                clauses.append((gate2lit[gate.left],                         -gate2lit[gate]))  # noqa
                clauses.append((                       gate2lit[gate.right], -gate2lit[gate]))  # noqa
            elif isinstance(gate, A.aig.Input):
                if gate.name in states:      # We already have it from init or end of last round
                    continue
                else:                 # This is a real output, add and remember it
                    action_name = '{}_{}'.format(gate.name,time)
                    in2lit[action_name] = gate2lit[gate]
        outlits.extend([gate2lit[step.aig.node_map[o]] for o in circ.aig.outputs])
        for s in states:
            assert step.aig.node_map[s] in gate2lit.keys()
            gate2lit[A.aig.Input(s)] = gate2lit[step.aig.node_map[s]]
    
    if truth_strategy == 'all':
        for lit in outlits:
            clauses.append((lit,))
    elif truth_strategy == 'last':
        clauses.append((outlits[-1],))
    else:
        raise "Help!"

    return ACNF.cnf.CNF(clauses, in2lit, outlits, None)

class GridSampler(SamplerBase):
  def __init__(self, size=8, horizon=2,  gridinfo=False, **kwargs):
    SamplerBase.__init__(self, **kwargs)
    self.size = int(size)
    self.horizon = int(horizon)
    self.gridinfo = gridinfo
    self.X = BV.atom(self.size, 'x', signed=False)
    self.Y = BV.atom(self.size, 'y', signed=False)
    self.mask_test = get_mask_test(self.X, self.Y)

  def encode_state(x, y):
    x, y = [BV.encode_int(self.size, 1 << (v - 1), signed=False) for v in (x, y)]
    return {'x': tuple(x), 'y': tuple(y)}


  def make_spec(self):
    LAVA, RECHARGE, WATER, DRY = map(LTL.atom, ['red', 'yellow', 'blue', 'brown'])

    EVENTUALLY_RECHARGE = RECHARGE.once()
    AVOID_LAVA = (~LAVA).historically()

    RECHARGED_AND_ONCE_WET = RECHARGE & WATER.once()
    DRIED_OFF = (~WATER).since(DRY)

    DIDNT_RECHARGE_WHILE_WET = (RECHARGED_AND_ONCE_WET).implies(DRIED_OFF)
    DONT_RECHARGE_WHILE_WET = DIDNT_RECHARGE_WHILE_WET.historically()

    CONST_TRUE = LTL.atom(True)


    SPECS = [
      CONST_TRUE, AVOID_LAVA, EVENTUALLY_RECHARGE, DONT_RECHARGE_WHILE_WET,
      AVOID_LAVA & EVENTUALLY_RECHARGE & DONT_RECHARGE_WHILE_WET,
      AVOID_LAVA & EVENTUALLY_RECHARGE,
      AVOID_LAVA & DONT_RECHARGE_WHILE_WET,
      EVENTUALLY_RECHARGE & DONT_RECHARGE_WHILE_WET,
    ]

    SPEC_NAMES = [
      "CONST_TRUE", "AVOID_LAVA", "EVENTUALLY_RECHARGE", "DONT_RECHARGE_WHILE_WET",
      "AVOID_LAVA & EVENTUALLY_RECHARGE & DONT_RECHARGE_WHILE_WET",
      "AVOID_LAVA & EVENTUALLY_RECHARGE",
      "AVOID_LAVA & DONT_RECHARGE_WHILE_WET",
      "EVENTUALLY_RECHARGE & DONT_RECHARGE_WHILE_WET",
    ]

    return AVOID_LAVA
    # return {k: v for (v,k) in zip(SPECS,SPEC_NAMES)}

  def get_feature_mask(self, feat, world, pairs_only=False):
    size = world.shape[1]
    mask_func = self.mask_test
    feat_rows = [feat in x for x in world]
    if np.sum([feat in x for x in world.transpose()]) < np.sum(feat_rows):
      feat_mask_tuples= self.get_feature_mask(feat,world.transpose(), pairs_only=True)
      flip = lambda f: lambda *a: f(*reversed(a))
      mask_func = flip(mask_func)
    else:
      def to_int(row, size):
        return int(sum([2**i for (x,i) in zip(reversed(row),np.arange(size)) if x]))
        
      feat_mask_tuples = []
      for (x,row) in zip(np.where(feat_rows)[0]+1, (world[feat_rows]==feat)):
        row_mask = (int(1 << x-1), to_int(row, size))
        feat_mask_tuples.append(row_mask)
    if pairs_only:
      return feat_mask_tuples
    if not len(feat_mask_tuples):
      return mask_func(0,0)
    else:   # Return the actual circuit for the mask
      return functools.reduce(lambda x,y: x | y,fn.map(lambda tup: mask_func(*tup), feat_mask_tuples))


  def get_random_masks(self, seed):
    world = world_from_noise(get_noise_grid(shape=(self.size, self.size),base=seed, persistence=5.0, lacunarity=2.0, scale=100))
    random_aps = {
      'yellow': self.get_feature_mask(1, world), 'red': self.get_feature_mask(3,world), 
      'blue': self.get_feature_mask(4,world), 'brown': self.get_feature_mask(2,world)
    }
    
    return random_aps
  


  def make_grid(self, seed=None):
    if not seed:
      seed = int(time.time())+os.getpid()
    random.seed(seed)
    base = random.randint(1,2**16)
    x = random.randint(1,self.size)
    y = random.randint(1,self.size)
    DYN = GW.gridworld(self.size, start=(x, y), compressed_inputs=True)
    APS = self.get_random_masks(base)
    # APS = {       #            x-axis       y-axis
    #   'yellow': self.mask_test(0b1000_0001, 0b1000_0001),
    #   'blue':   self.mask_test(0b0001_1000, 0b0011100),
    #   'brown':   self.mask_test(0b0011_1100, 0b1000_0001),
    #   'red':    self.mask_test(0b1000_0001, 0b0100_1100) \
    #           | self.mask_test(0b0100_0010, 0b1100_1100),
    # }

    SENSOR = create_sensor(APS)
    spec = self.make_spec()
    MONITOR = spec2monitor(spec)
    circuit = DYN >> SENSOR >> MONITOR
    return mdp2cnf(circuit,self.horizon+random.choice([-1,0,1])), seed
    
  def sample(self, stats_dict: dict) -> (FileName, FileName):
    fcnf, seed = self.make_grid()
    name = '{}_{}.cnf'.format(random_string(16),seed+os.getpid())
    fname = '/tmp/{name}.cnf'

    self.write_expression(fcnf, fname, is_cnf=True)
    return fname, None

