import ipdb
from tick import *
clock = GlobalTick()

def break_every_tick(n):
  t = clock.get_tick()
  if (t % n) == 0 and t > 0:
    ipdb.set_trace()
