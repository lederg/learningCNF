import ipdb
from sacred import Experiment
from config import *
from settings import *

settings = CnfSettings(cfg())
ex = Experiment('REINFORCE')
ex.add_config(settings.hyperparameters)

@ex.automain
def main():	
	settings.hyperparameters['name'] = ex.current_run.experiment_info['name']
	from task_cadet import cadet_main
	cadet_main()

