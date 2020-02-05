import os
import ipdb
import dgl
import torch
import torch.nn as nn
import torch.optim as optim

from settings import *
from clause_model import *
from supervised_cnf_dataset import *
from ray import tune
from ray.experimental.sgd.pytorch import utils as pytorch_utils
from ray.experimental.sgd.utils import TimerStat
from ray.experimental.sgd.pytorch.pytorch_trainer import PyTorchTrainer, PyTorchTrainable
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.utils import validate_save_restore
from ray.tune.trial import ExportFormat

from tqdm import tqdm
from pprint import pprint

def train(model, train_iterator, criterion, optimizer, config):
  """Runs 1 training epoch"""
  print('Beginning epoch')
  if isinstance(model, collections.Iterable) or isinstance(
      optimizer, collections.Iterable):
    raise ValueError(
        "Need to provide custom training function if using multi-model "
        "or multi-optimizer training.")

  batch_time = pytorch_utils.AverageMeter()
  data_time = pytorch_utils.AverageMeter()
  losses = pytorch_utils.AverageMeter()
  labels_bias = pytorch_utils.AverageMeter()

  timers = {k: TimerStat() for k in ["d2h", "fwd", "grad", "apply"]}
  correct = 0
  total = 0
  # switch to train mode
  model.train()

  end = time.time()
  for (features, target) in tqdm(train_iterator):
  # for (features, target) in train_iterator:
    # measure data loading time
    data_time.update(time.time() - end)
    labels_bias.update(target.sum().float() / target.size(0), target.size(0))
    # Create non_blocking tensors for distributed training
    with timers["d2h"]:
      if torch.cuda.is_available():
        features = features.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

    # compute output
    with timers["fwd"]:
      output = model(features)
      loss = criterion(output, target)
      _, predicted = torch.max(output.data, 1)
      total += target.size(0)
      correct += (predicted == target).sum().item()
      # print(output)
      # print('predicted variability: {}/{}'.format(predicted.sum().float(),predicted.size(0)))
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
    "train_accuracy": correct/total,
    "labels_bias": labels_bias.avg,
    "batch_time": batch_time.avg,
    "batch_processed": losses.count,
    "train_loss": losses.avg,
    "data_time": data_time.avg,
  }
  stats.update({k: t.mean for k, t in timers.items()})
  print('train(): Stats are:')
  pprint(stats)
  return stats

def validate(model, val_iterator, criterion, config):
  print('Beginning validation')
  if isinstance(model, collections.Iterable):
    raise ValueError(
      "Need to provide custom validation function if using multi-model "
      "training.")
  batch_time = pytorch_utils.AverageMeter()
  losses = pytorch_utils.AverageMeter()
  labels_bias = pytorch_utils.AverageMeter()

  # switch to evaluate mode
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    end = time.time()
    for (features, target) in tqdm(val_iterator):
      labels_bias.update(target.sum().float() / target.size(0), target.size(0))
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

  stats = {"labels_bias": labels_bias.avg, "batch_time": batch_time.avg, "validation_loss": losses.avg}
  stats.update(mean_accuracy=correct / total)
  print('validate(): Stats are:')
  pprint(stats)  
  return stats

def update_settings(config):
  settings = CnfSettings()
  if 'settings' in config:
    new_settings = config['settings']
    for k in new_settings.keys():
      if k in config.keys():
        settings.hyperparameters[k] = config[k]
      else:        
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
  ds = CnfGNNDataset(settings['rl_train_data'], transform=transforms.Compose([SampleLearntClauses(10),CapActivity()]))
  validate_ds = CnfGNNDataset(settings['rl_validation_data'], transforms.Compose([SampleLearntClauses(10),CapActivity()]))
  rc1 = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=sat_collate_fn,shuffle=True, num_workers=3)
  rc2 = torch.utils.data.DataLoader(validate_ds, batch_size=batch_size, collate_fn=sat_collate_fn,shuffle=True, num_workers=2)
  return rc1, rc2

def clause_prediction_main():
  settings = CnfSettings()
  import ray
  address = settings['ray_address']
  if address:
    print('Running in ray cluster mode')
    run.init(address=address, redis_password='blabla')
  else:
    ray.init(num_cpus=settings['parallelism']+1)
  # criterion = torch.nn.CrossEntropyLoss()
  # model = model_creator({})
  # optimizer = optimizer_creator(model, {'lr': 1e-2})
  # train_loader, validation_loader = data_creator(settings['batch_size'], {})
  cross_loss = lambda x: nn.CrossEntropyLoss()
  config = {    
    "model_creator": tune.function(model_creator),
    "data_creator": tune.function(data_creator),
    "optimizer_creator": tune.function(optimizer_creator),
    "loss_creator": tune.function(cross_loss),
    "train_function": tune.function(train),
    "validation_function": tune.function(validate),
    "num_replicas": settings['parallelism'],
    "use_gpu": False,
    "batch_size": settings['batch_size'],
    "config": {
      "lr": settings['init_lr'],
      # "max_iters": tune.grid_search([1,2,3]),
      # "lr": tune.grid_search([1e-2,settings['init_lr']]),
      "settings": settings.hyperparameters,
      },
  }

  pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="mean_accuracy",
    mode="max",
    perturbation_interval=6,
    hyperparam_mutations={
        # distribution for resampling
        "lr": lambda: random.uniform(0.0001, 0.02),
    })


  analysis = tune.run(
    PyTorchTrainable,
    name=settings['name'],
    num_samples=4,
    scheduler=pbt,
    reuse_actors=True,    
    config=config,
    stop={"training_iteration": 400},
    verbose=1)

  rc = analysis.get_best_config(metric="validation_loss", mode="min")
  print('Finished. Printing Analysis:\n\n\n\n')
  print(rc)

  # trainer1 = PyTorchTrainer(
  #   model_creator,
  #   data_creator,
  #   optimizer_creator,
  #   loss_creator=cross_loss,
  #   train_function=train,
  #   validation_function=validate,
  #   config={'lr': 1e-2, 'settings': settings.hyperparameters},
  #   num_replicas=settings['parallelism'],
  #   # use_gpu=True,
  #   batch_size=settings['batch_size'],
  #   )
  # for i in range(settings['num_epochs']):
  #   stats = trainer1.train()  
  #   # stats = train(model,train_loader, criterion, optimizer, {}) 
  #   print(stats)
  #   if i>0 and i%2 == 0:
  #     stats = trainer1.validate()
  #     print(stats)
  # 