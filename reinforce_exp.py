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
	# print(settings.hyperparameters)
	from task_cadet import cadet_main
	if not settings['do_not_run']:
		cadet_main()

