import os
import random
import numpy as np
import itertools
from gen_utils import random_string

# (define (problem bw_5_1)
#   (:domain blocks-domain)
#   (:objects b1 b2 b3 b4 b5 - block)
#   (:init (emptyhand) (on b1 b3) (on b2 b1) (on-table b3) (on-table b4) (on b5 b4) (clear b2) (clear b5))
#   (:goal (and (emptyhand) (on b1 b2) (on b2 b5) (on-table b3) (on-table b4) (on-table b5) (clear b1) (clear b3) (clear b4)))
# )
def sample_config(n):
    random.seed()
    sizeofstart=random.randint(1,n)
    start=[]
    for i in range(0,sizeofstart):
        start.append(['a',])
    for i in range(0,n):
        x=random.randint(0,sizeofstart-1)
        start[x].append(i)
        if 'a' in start[x]:
            start[x].remove('a')
    fstart=[]
    for i in start:
        if 'a' not in i:
            fstart.append(tuple(i))
    tuplestart=tuple(fstart)
    return tuplestart

    # sizeofend=random.randint(1,n)
    # end=[]
    # for i in range(0,sizeofend):
    #     end.append(['a',])
    # for i in range(0,n):
    #     x=random.randint(0,sizeofend-1)
    #     end[x].append(i)
    #     if 'a' in end[x]:
    #         end[x].remove('a')
    # fend=[]
    # for i in end:
    #     if 'a' not in i:
    #         fend.append(tuple(i))
    # tupleend=tuple(fend)
    # problem=(tuplestart,tupleend,)
    # return problem

def shuffle(config, steps_max):
    prev_config = config
    steps = 0
    attempts = 0
    while steps < steps_max:
        new_config = list(config)
        s = random.randint(0, len(new_config) - 1)
        d = random.randint(0, len(new_config))
        # print(new_config)
        # print(s)
        source = new_config.pop(s)
        # print("sUSSS")
        elem = source[-1]
        source = source[:-1]
        if (source != ()):
            new_config += [source]
        destin = new_config.pop(d) if (d<len(new_config)) else ()
        destin = tuple(list(destin) + [elem])
        new_config += [destin]
        new_config = tuple(new_config)

        attempts += 1
        if (sorted(new_config) != sorted(prev_config) and sorted(new_config) != sorted(config)):
            prev_config = config
            config = new_config
            # print(steps, config)
            steps += 1

        if (attempts == 100):
            break
    return config

# if __name__ == '__main__':
#     a = sample_config(5)
#     print("init", a)
#     print("end", shuffle(a, 2))
#     # print(shuffle(a, 1))


def to_pddl(config):
    res = ""
    for stack in config:
        for i, item in enumerate(stack):
            if (i == 0): # The first item (i.e., on the table)
                res += f" (on-table b{item})"
            else:
                res += f" (on b{item} b{stack[i-1]})"

            if (i == len(stack) - 1): # Last item on the stack (clear)
                res += f" (clear b{item})"

    return "(emptyhand)" + res

def gen(size, steps, fname, rand_id):
    # print("size", size)
    task = open(fname, 'w')

    task.write(f"(define (problem bw_{size}_{rand_id})")
    task.write('\n')
    task.write(f"\t(:domain blocks-domain)")
    task.write('\n')
    task.write(f"\t(:objects {' '.join(['b'+str(i)  for i in range(size)])} - block)")
    task.write('\n')

    init = sample_config(size)
    goal = shuffle(init, steps)
    # print("init ", init)
    # print("goal ", goal)
    task.write(f"\t(:init {to_pddl(init)})")
    task.write('\n')
    task.write(f"\t(:goal (and {to_pddl(goal)}))")
    task.write('\n')

    task.write(')')
    task.write('\n')
    task.close()

class BlocksGen():
    def __init__(self, size_min=5, size_max=8, steps_min=1, steps_max=4, **kwargs):
        size_range = range(int(size_min), int(size_max))
        steps_range = range(int(steps_min), int(steps_max))

        self.ranges = list(itertools.product(size_range, steps_range))

    def sample(self):
        rand_id = random_string(8)
        (size, step) = random.choice(self.ranges)
        problem_id = "blocks-%i-%i-%s" % (size, step, rand_id)
        fname = os.path.join("/tmp", f"{problem_id}.pddl")
        gen(size, step, fname, rand_id)

        return problem_id, fname
