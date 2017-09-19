import re
import ipdb
from pymongo import MongoClient
from model import *
from settings import *
from datautils import *


def test_model_from_file(fname):
    p = re.compile('^.*run_([a-zA-Z0-9_]*)_nc([0-9]*)(.*)__([0-9]*).model')
    m = p.match(fname)
    nc = m.group(2)
    params, dmode = load_hyperparams(m.group(1),int(m.group(4)))
    if not 'data_mode' in params:
    	params['data_mode'] = dmode
    settings = CnfSettings(params)
    settings['num_classes'] = int(nc)
    net = load_model_from_file

def load_model_from_file(**kwargs):
	settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
    model_class = eval(settings['classifier_type'])
    net = model_class(**settings.hyperparameters)
    return net

def load_hyperparams(name, time):
    with MongoClient() as client:
        db = client['graph_exp']
        runs = db['runs']        
        rc = runs.find_one({'experiment.name': name, 'config.hyperparams.time': time})
        g = rc['config']['data_mode'].values()
        dmode = list(list(g)[0][1].values())[0][0]
        return rc['config']['hyperparams'], dmode


