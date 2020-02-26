import os
import ipdb
import psutil
import tracemalloc
import dgl
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from settings import *
from clause_model import *
from supervised_cnf_dataset import *
from ray import tune
from ray.experimental.sgd.pytorch import utils as pytorch_utils
from ray.experimental.sgd.utils import TimerStat
from ray.experimental.sgd.pytorch.pytorch_trainer import PyTorchTrainer
from cp_trainable import ClausePredictionTrainable
from ray.tune.schedulers import PopulationBasedTraining

from tqdm import tqdm
from pprint import pprint

def initialization_hook(runner):
  print('initialization_hook!!')
  print(os.environ)

def train(model, train_iterator, criterion, optimizer, config):
  """Runs 1 training epoch"""
  settings = update_settings(config)  
  if settings['memory_profiling']:
    tracemalloc.start(25)
  main_proc = psutil.Process(os.getpid())
  print('Beginning epoch')
  utils.set_lr(optimizer,settings['init_lr'])
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
        features = cudaize_sample(features)
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

    if settings['memory_profiling']:    
      try:
        total_mem = main_proc.memory_info().rss / float(2 ** 20)
        children = main_proc.children(recursive=True)
        for child in children:
          child_mem = child.memory_info().rss / float(2 ** 20)
          total_mem += child_mem
          print('Child pid is {}, name is {}, mem is {}'.format(child.pid, child.name(), child_mem))
        print('Total memory on host is {}'.format(total_mem))
        print(" Host RSS memory: {0:.2f}MB".format(self.process.memory_info().rss / float(2 ** 20)))        
        objects = gc.get_objects()
        print('Number of objects is {}'.format(len(objects)))
        del objects
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("[ Top 20]")
        for stat in top_stats[:20]:
            print(stat)
        del snapshot
        del top_stats

      except:       # A child could already be dead due to a race. Just ignore it this round.
        print('why like this')


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
        features = cudaize_sample(features)
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

# update_settings sometimes gets all settings from config (in cluster fork case), or just updates


def update_settings(config):
  settings = CnfSettings()
  if 'settings' in config:
    settings.hyperparameters = config.pop('settings')
  for k in config.keys():
    settings.hyperparameters[k] = config[k]
  return settings


def model_creator(config):
  settings = update_settings(config)
  return ClausePredictionModel(settings)

def optimizer_creator(model, config):
  settings = update_settings(config)  
  return torch.optim.SGD(model.parameters(), lr=settings['init_lr'])

def data_creator(batch_size, config):
  settings = update_settings(config)
  cmask_features = settings['cp_cmask_features']
  vmask_features = settings['cp_vmask_features']
  if settings['cp_invert_cmask']:
    cmask_features = set(range(6)).difference(cmask_features)
  if settings['cp_invert_vmask']:
    vmask_features = set(range(6)).difference(vmask_features)
  trans = []
  if settings['cp_task'] == 'ever_used':
    trans.append(MakeEverUsedTarget())
  elif settings['cp_task'] == 'lbd':
    trans.append(MakeLbdTarget())
  trans.extend([CapActivity(),SampleLearntClauses(settings['cp_clauses_sampled'],settings['cp_num_categories'])])
  trans.extend([ZeroClauseIndex(x) for x in cmask_features] + [ZeroLiteralIndex(x) for x in vmask_features])
  if settings['cp_cap_graph']:
    trans.append(CapGraph((settings['max_iters']-1)*2+1))
  trans = transforms.Compose(trans)
  ds = CnfGNNDataset(settings['rl_train_data'], transform=trans)
  validate_ds = CnfGNNDataset(settings['rl_validation_data'], transform=trans)
  rc1 = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=sat_collate_fn,shuffle=True, num_workers=3)
  rc2 = torch.utils.data.DataLoader(validate_ds, batch_size=batch_size, collate_fn=sat_collate_fn,shuffle=True, num_workers=2)
  return rc1, rc2

def clause_prediction_main():
  settings = CnfSettings()
  import ray
  address = settings['ray_address']
  if address:
    print('Running in ray cluster mode')
    ray.init(address=address, redis_password='blabla')
  else:
    ray.init()
    # ray.init(num_cpus=settings['parallelism']+1)
  # criterion = torch.nn.CrossEntropyLoss()
  cross_loss = lambda x: nn.CrossEntropyLoss()
  restore_point = settings['base_model']
  if not settings['smoke_test']:
    config = {    
      "model_creator": model_creator,
      "data_creator": data_creator,
      "optimizer_creator": optimizer_creator,
      "loss_creator": cross_loss,
      "train_function": train,
      "validation_function": validate,
      # "initialization_hook": initialization_hook,
      "num_replicas": settings['parallelism'],
      "use_gpu": False,
      "batch_size": settings['batch_size'],
      "config": {
        # "init_lr": settings['init_lr'],
        # "lr": tune.grid_search([1e-2,settings['init_lr']]),
        # "max_iters": tune.grid_search([0,1,2,3,4]),
        # "use_sum": tune.grid_search([True, False]),
        # "non_linearity": tune.grid_search(['torch.tanh', 'torch.relu']),
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
      ClausePredictionTrainable,
      name=settings['name'],
      num_samples=settings['cp_num_samples'],
      # scheduler=pbt,
      checkpoint_freq=settings['cp_save_every'],
      restore=restore_point,
      reuse_actors=True,    
      resources_per_trial={'cpu': 6, 'memory': 2**33},
      config=config,
      stop={"training_iteration": 120},
      verbose=1)

    rc = analysis.get_best_config(metric="validation_loss", mode="min")
    print('Finished. Printing Analysis:\n\n\n\n')
    print(rc)

  else:
    # model = model_creator({})
    # optimizer = optimizer_creator(model, {'settings': settings})
    # train_loader, validation_loader = data_creator(settings['batch_size'], {})

    trainer1 = PyTorchTrainer(
      model_creator,
      data_creator,
      optimizer_creator,
      loss_creator=cross_loss,
      train_function=train,
      validation_function=validate,
      config={'settings': settings.hyperparameters},
      num_replicas=settings['parallelism'],
      use_gpu=settings['cp_use_gpu'],
      batch_size=settings['batch_size'],
      )
    if restore_point:
      trainer1.restore(restore_point)


    for i in range(settings['num_epochs']):
      stats = trainer1.train()  
      # stats = train(model,train_loader, criterion, optimizer, {}) 
      print(stats)
      if i>0 and i%2 == 0:
        stats = trainer1.validate()
        print(stats)
    