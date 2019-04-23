import ipdb
import cProfile
from sacred import Experiment
from config import *
from settings import *

# settings = CnfSettings(cfg())
# ex.add_config(settings.hyperparameters)


@ex.automain
def main():	
	settings = CnfSettings(ex.current_run.config)
	settings.hyperparameters['name'] = ex.current_run.experiment_info['name']
	settings.hyperparameters['mp']=True
	# print(settings.hyperparameters)
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
