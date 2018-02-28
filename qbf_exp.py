from config import *
from sacred import Experiment
from qbf_data import *
from rl_model import *

settings = CnfSettings(cfg())
ex = Experiment('QBF')
ex.add_config(settings.hyperparameters)

@ex.automain
def main():
	from task_qbf_train import qbf_train_main
	settings.hyperparameters['name'] = ex.current_run.experiment_info['name']
	qbf_train_main()
