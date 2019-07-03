import ipdb
import cProfile
import logging
import json
from sacred import Experiment
from sacred.observers import MongoObserver
from config import *
from settings import *

# settings = CnfSettings(cfg())
# ex.add_config(settings.hyperparameters)


@ex.automain
def main():	
	if ex.current_run.config['update_config']:
		with open(ex.current_run.config['update_config']) as fp:
			conf = json.load(fp)
			for k in conf.keys():
				ex.current_run.config[k] = conf[k]
			ex.add_config(conf)
	settings = CnfSettings(ex.current_run.config)
	settings.hyperparameters['name'] = ex.current_run.experiment_info['name']
	settings.hyperparameters['mp']=True
	logging.basicConfig(level=logging.DEBUG)
	if settings['mongo']:
		print('Adding mongodb observer')
		ex.observers.append(MongoObserver.create(url=settings['mongo_host'],
                                         db_name=settings['mongo_dbname']))
	if settings['save_config']:
		with open('config_{}.json'.format(settings['name']), 'w') as fp:
			json.dump(settings.hyperparameters, fp, indent=4)
	from task_a3c import a3c_main
	from task_parallel import parallel_main
	from task_collectgrid import grid_main
	from task_collectrandom import collect_random_main
	from task_lbd import collect_lbd_main
	from task_cadet import cadet_main

	func = eval(settings['main_loop'])
	if settings['profiling']:
		cProfile.runctx(settings['main_loop']+'()', globals(), locals(), 'main_process.prof')
	else:
		func()
