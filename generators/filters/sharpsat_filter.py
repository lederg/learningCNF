import os

from filters.filter_base import FilterBase
from gen_types import FileName

from pysat.solvers import SharpSAT

"""
Filters the shaprSAT instance according to a set of constrainst:
    steps_min
    time_min
    time_max

sample usage:
>>> # import SharpSATFilter
>>> file_path = # path to the cnf file
>>> time_max = 2
>>> filter = SharpSATFilter()
>>> filter.filter(file_path)
"""
class SharpSATFilter(FilterBase):
    def __init__(self, steps_min = 30, time_min = 0.15, time_max = 2, **kwargs):
        FilterBase.__init__(self, **kwargs)
        self.steps_min = int(steps_min)
        self.time_min = float(time_min)
        self.time_max = int(time_max)

    def filter(self, fname: FileName, stats_dict: dict) -> bool:
        # Check for degenerate (because sharpSAT's pysat port does not support CNF object yet)
        with open(fname, 'r') as f:
            z = f.readline()
            if z.startswith('p cnf 0 1'):
                self.log.info('{}: degenerate'.format(fname))
                return False


        sharpSAT = SharpSAT(time_budget = self.time_max, use_timer= True)
        count = sharpSAT.solve(fname)

        message = ""
        res = True
        if (sharpSAT.reward() < self.steps_min):
            message = f"{fname}: Too easy! Steps < {self.steps_min}"
            res = False
        if (sharpSAT.time() < self.time_min):
            message = f"{fname}: Too easy! Time < {self.time_min}s"
            res = False
        if (self.time_max <= sharpSAT.time()):
            message = f"{fname}: Too hard! Time > {self.time_max}s"
            res = False

        if (res): # Instance accepted. Save the stats about the new instance
            message = f"{fname}: Accepted"
            stats_dict.update({
                'var_len': sharpSAT.nof_vars(),
                'cla_len': sharpSAT.nof_clauses(),
                'op_cnt' : sharpSAT.reward(),
                'time'   : f"{sharpSAT.time():.2f}",
                'model_cnt' : count
            })

        message += f" (step/time/pid: {sharpSAT.reward()}/{sharpSAT.time():.2f}/{os.getpid()})"
        self.log.info(message)
        # self.log.info(f"{fname}: {res}")

        return res
