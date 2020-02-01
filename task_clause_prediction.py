import os
import ipdb
import dgl
import torch
import torch.nn as nn
import torch.optim as optim

from settings import *
from clause_model import *
from supervised_cnf_dataset import *
from ray.experimental.sgd.pytorch import utils as pytorch_utils
from ray.experimental.sgd.utils import TimerStat
from tqdm import tqdm

def train(model, train_iterator, criterion, optimizer, config):
  """Runs 1 training epoch"""
  if isinstance(model, collections.Iterable) or isinstance(
      optimizer, collections.Iterable):
    raise ValueError(
        "Need to provide custom training function if using multi-model "
        "or multi-optimizer training.")

  batch_time = pytorch_utils.AverageMeter()
  data_time = pytorch_utils.AverageMeter()
  losses = pytorch_utils.AverageMeter()

  timers = {k: TimerStat() for k in ["d2h", "fwd", "grad", "apply"]}

  # switch to train mode
  model.train()

  end = time.time()

  for (features, target) in tqdm(train_iterator):
    # measure data loading time
    data_time.update(time.time() - end)

    # Create non_blocking tensors for distributed training
    with timers["d2h"]:
      if torch.cuda.is_available():
        features = features.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

    # compute output
    with timers["fwd"]:
      output = model(features)
      loss = criterion(output, target)
      # measure accuracy and record loss
      losses.update(loss.item(), output.size(0))

    with timers["grad"]:
      # compute gradients in a backward pass
      optimizer.zero_grad()
      loss.backward()

    with timers["apply"]:
      # Call step of optimizer to update model params
      optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

  stats = {
    "batch_time": batch_time.avg,
    "batch_processed": losses.count,
    "train_loss": losses.avg,
    "data_time": data_time.avg,
  }
  stats.update({k: t.mean for k, t in timers.items()})
  return stats

def clause_prediction_main():
	settings = CnfSettings()
	criterion = torch.nn.CrossEntropyLoss()
	ds = CnfGNNDataset(settings['rl_train_data'], transform=CapActivity())
	model = ClausePredictionModel()
	dataloader = torch.utils.data.DataLoader(ds, batch_size=settings['batch_size'], collate_fn=sat_collate_fn,shuffle=True, num_workers=3)
	optimizer = optim.SGD(model.parameters(), lr=settings['init_lr'], momentum=0.9)
	rc = train(model,dataloader, criterion, optimizer, {})	
	ipdb.set_trace()
	