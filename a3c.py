import ipdb
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
	if settings['main_loop'] == 'a3c':		
		a3c_main()
	elif settings['main_loop'] == 'parallel':
		parallel_main()
	else:
		print('Unknown main loop: {}'.format(settings['main_loop']))
