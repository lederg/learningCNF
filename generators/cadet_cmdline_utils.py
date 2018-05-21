import os
import re
import ipdb
import numpy as np
import tempfile
from subprocess import Popen, PIPE, STDOUT

from aux_utils import is_number


def extract_num_conflicts(s):
    res = re.findall(' Conflicts: (\d+)', str(s))
    if len(res) == 1:
        return int(res[0])
    else:
        print('  ERROR: {}'.format(s))
        return 0


def extract_num_decisions(s):
    res = re.findall(' Decisions: (\d+)', str(s))
    if len(res) == 1:
        return int(res[0])
    else:
        print('  ERROR: {}'.format(s))
        return 0


def eval_formula(formula, repetitions=1, decision_limit=0):
    assert isinstance(formula, str)
    assert isinstance(decision_limit, int)

    returncodes = []
    conflicts = []
    decisions = []

    f = tempfile.NamedTemporaryFile()
    f.write(formula.encode())
    f.seek(0)

    for _ in range(repetitions):
        tool = ['./../../cadet/dev/cadet','-v','1',
                '--debugging',
                '--cegar_soft_conflict_limit',
                '-l', f'{decision_limit}',
                '--sat_by_qbf',
                '--random_decisions',
                '--fresh_seed',
                f.name]
        p = Popen(tool, stdout=PIPE, stdin=PIPE)
        stdout, stderr = p.communicate()

        if p.returncode not in [10, 20, 30]:
            print(stdout)
            print(stderr)
            quit()
        # print('  ' + str(p.returncode))
        # print('  ' + str(stdout))
        # print('  ' + str(stderr))
        
        # print('  Maxvar: ' + str(maxvar))
        # print('  Conflicts: ' + str(extract_num_conflicts(stdout)))
        if p.returncode == 30:
            f.close()
            return 30, 0.0, 0.0
        returncodes.append(p.returncode)
        conflicts.append(extract_num_conflicts(stdout))
        num_decisions = extract_num_decisions(stdout)
        decisions.append(num_decisions)

        if decision_limit != 0 and num_decisions > decision_limit:
            print('Error: decision limit was violated')
            print(formula)
            print(' '.join(tool))
            print(stdout)
            quit()

    f.close()

    assert all(x == returncodes[0] for x in returncodes)
    return returncodes[0], np.mean(conflicts), np.mean(decisions)

