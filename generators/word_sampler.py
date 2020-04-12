import os
import tempfile
from sampler_base import *
from random import randint, seed
from aigerbv.aigbv import AIGBV
from aigerbv.expr import SignedBVExpr, atom
import aiger_analysis as aa

word_length = 8

arith_ops = [SignedBVExpr.__add__, SignedBVExpr.__sub__]
bitwise_ops = [SignedBVExpr.__and__, SignedBVExpr.__or__, SignedBVExpr.__xor__]
unary_ops = [SignedBVExpr.__invert__, SignedBVExpr.__abs__, SignedBVExpr.__neg__]
cmp_ops = [SignedBVExpr.__eq__, SignedBVExpr.__ne__,
           SignedBVExpr.__lt__, SignedBVExpr.__le__, SignedBVExpr.__gt__, SignedBVExpr.__ge__]

variables = None


def variable():
  # variables[randint(0, len(variables) - 1)]
  global variables
  return atom(word_length, variables.pop(), signed=True)


def constant_expr():
  if randint(0, 4) == 0:
    c = atom(word_length, randint(- 2**(word_length-1), 2**(word_length-1) - 1), signed=True)
  else: 
    c = atom(word_length, 1, signed=True)
  return c


def leaf_expr(size):  # constant or variable
  assert size == 1
  return variable() if randint(0, 1) == 0 else constant_expr()


def unary_expr(size):
  assert size > 1
  arg = random_expr(size - 1)
  op = unary_ops[randint(0, len(unary_ops) - 1)]
  return op(arg)


def arithmetic_expr(size):
  assert size > 2
  operator = arith_ops[randint(0, len(arith_ops)-1)]
  # arg_num = len(inspect.getargspec(operator)[0])  # arity of the function
  split = randint(1, size - 2)  # at least one operation on either side
  left = random_expr(split)
  right = random_expr(size - split - 1)
  return operator(left, right)


def random_expr(size):
  if size <= 1:
    return leaf_expr(size)
  if size == 2:
    return unary_expr(size)
  if randint(0, 2) == 0:
    return bitwise_expr(size)
  else:
    return arithmetic_expr(size)


def bitwise_expr(size):
  assert size > 2
  op = bitwise_ops[randint(0, len(bitwise_ops)-1)]
  split = randint(1, size - 2)  # at least one operation on either side
  left = random_expr(split)
  right = random_expr(size - split - 1)
  return op(left, right)

def random_bool_expr(size):
  global variables
  variables = ['2 y1', '1 x1', '2 y2', '1 x2']

  assert size > 2
  op = cmp_ops[randint(0, len(cmp_ops)-1)]
  split = randint(1, size - 2)  # at least one operation on either side
  left = random_expr(split)
  right = random_expr(size - split - 1)
  return op(left, right)


def random_circuit(size):
  while True:
    e = random_bool_expr(size)
    e = aa.simplify(e)
    if e is not None:
      return e
    else: 
      print('    Failed to generate expression; trying again')

class WordSampler(SamplerBase):
  def __init__(self, config):
    self.size = int(config.get('size',5))
    self.dir = config.get('dir', '.')
  def sample(self):
    e = random_circuit(self.size)
    f = tempfile.NamedTemporaryFile()
    f.write(str(e).encode())
    f.seek(0)
    rc = f.name+'.cnf'
    os.system('aigtocnf {} {}'.format(f.name,rc))
    return rc

