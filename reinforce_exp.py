import ipdb
from sacred import Experiment
from config import *
from settings import *

settings = CnfSettings(cfg())
ex = Experiment('REINFORCE')
ex.add_config(settings.hyperparameters)

@ex.automain
def main():	
	from task_cadet import cadet_main
	cadet_main()

