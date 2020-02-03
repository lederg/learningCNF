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
from ray.experimental.sgd.pytorch.pytorch_trainer import PyTorchTrainer
from tqdm import tqdm

EPOCHS = 10

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
  # for (features, target) in train_iterator:
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

def validate(model, val_iterator, criterion, config):
  if isinstance(model, collections.Iterable):
    raise ValueError(
      "Need to provide custom validation function if using multi-model "
      "training.")
  batch_time = AverageMeter()
  losses = AverageMeter()

  # switch to evaluate mode
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    end = time.time()
    for i, (features, target) in enumerate(val_iterator):

      if torch.cuda.is_available():
        features = features.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

      # compute output
      output = model(features)
      loss = criterion(output, target)
      _, predicted = torch.max(output.data, 1)
      total += target.size(0)
      correct += (predicted == target).sum().item()

      # measure accuracy and record loss
      losses.update(loss.item(), output.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

  stats = {"batch_time": batch_time.avg, "validation_loss": losses.avg}
  stats.update(mean_accuracy=correct / total)
  return stats

def update_settings(config):
	settings = CnfSettings()
	if 'settings' in config:
		new_settings = config['settings']
		for k in new_settings.keys():
			settings.hyperparameters[k] = new_settings[k]
	return settings


def model_creator(config):
	settings = update_settings(config)
	return ClausePredictionModel(settings)

def optimizer_creator(model, config):
	settings = update_settings(config)
	return torch.optim.SGD(model.parameters(), lr=config['lr'])

def data_creator(batch_size, config):
	settings = update_settings(config)
	ds = CnfGNNDataset(settings['rl_train_data'], transform=CapActivity())
	validate_ds = CnfGNNDataset(settings['rl_validation_data'], transform=CapActivity())
	rc1 = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=sat_collate_fn,shuffle=True, num_workers=3)
	rc2 = torch.utils.data.DataLoader(validate_ds, batch_size=batch_size, collate_fn=sat_collate_fn,shuffle=True, num_workers=2)
	return rc1, rc2

def clause_prediction_main():
	settings = CnfSettings()
	import ray
	ray.init()
	criterion = torch.nn.CrossEntropyLoss()
	model = model_creator({})
	optimizer = optimizer_creator(model, {'lr': 1e-2})
	train_loader, validation_loader = data_creator(settings['batch_size'], {})
	cross_loss = lambda x: nn.CrossEntropyLoss()
	trainer1 = PyTorchTrainer(
		model_creator,
		data_creator,
		optimizer_creator,
		loss_creator=cross_loss,
		train_function=train,
		validation_function=validate,
		config={'lr': 1e-2, 'settings': settings.hyperparameters},
		num_replicas=settings['parallelism'],
		# use_gpu=True,
		batch_size=settings['batch_size'],
		)


	for i in range(EPOCHS):
		stats = trainer1.train()	
		# stats = train(model,train_loader, criterion, optimizer, {})	
		print(stats)
	