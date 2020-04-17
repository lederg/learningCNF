import os
import random
import aiger as A
import aiger_bv as BV
import aiger_coins as C
import aiger_gridworld as GW
import aiger_ptltl as LTL
import aiger_cnf as ACNF
import funcy as fn
import matplotlib.pyplot as plt
import seaborn as sns

from generators.gentypes import FileName
from bidict import bidict
from sampler_base import SamplerBase
from random import randint, seed
from utils import *


def create_sensor(aps):
  sensor = BV.aig2aigbv(A.empty())
  for name, ap in aps.items():
      sensor |= ap.with_output(name).aigbv
  return sensor



def spec2monitor(spec):
  monitor = spec.aig | A.sink(['red', 'yellow', 'brown', 'blue'])
  monitor = BV.aig2aigbv(monitor)
  return monitor

class GridSampler(SamplerBase):
  def __init__(self, config):
    SamplerBase.__init__(self, config)    
    self.size = int(config.get('size',8))
    self.horizon = int(config.get('horizon',2))
    self.X = BV.atom(self.size, 'x', signed=False)
    self.Y = BV.atom(self.size, 'y', signed=False)

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
  def mask_test(self, xmask, ymask):
    return ((self.X & xmask) !=0) & ((self.Y & ymask) != 0)

  def make_grid(self):
    x = random.randint(1,self.size)
    y = random.randint(1,self.size)
    DYN = GW.gridworld(self.size, start=(x, y), compressed_inputs=True)
    APS = {       #            x-axis       y-axis
      'yellow': self.mask_test(0b1000_0001, 0b1000_0001),
      'blue':   self.mask_test(0b0001_1000, 0b0011100),
      'brown':   self.mask_test(0b0011_1100, 0b1000_0001),
      'red':    self.mask_test(0b1000_0001, 0b0100_1100) \
              | self.mask_test(0b0100_0010, 0b1100_1100),
    }

    SENSOR = create_sensor(APS)
    spec = self.make_spec()
    MONITOR = spec2monitor(spec)
    circuit = DYN >> SENSOR >> MONITOR
    unrolled_circuit = circuit.unroll(self.horizon)
    dongle = A.and_gate(unrolled_circuit.outputs)
    rc = unrolled_circuit >> BV.aig2aigbv(dongle)

    return rc
    
  def sample(self) -> FileName:
    e = self.make_grid()
    fname = '/tmp/{}.cnf'.format(random_string(16))
    self.write_expression(e, fname)
    return fname

