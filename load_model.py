import re
import ipdb
from pymongo import MongoClient
from batch_model import *
from settings import *
from datautils import *
from testing import test

def test_model_from_file(model_fname, test_fname=None):
    p = re.compile('^.*run_([a-zA-Z0-9_]*)_nc([0-9]*)(.*)__([0-9]*)_epoch[0-9]+.model')
    m = p.match(model_fname)
    nc = m.group(2)
    params, dmode, config = load_hyperparams(m.group(1),int(m.group(4)))
    dmode = DataMode(dmode)
    params['data_mode'] = dmode
    settings = CnfSettings(params)
    settings.hyperparameters['num_classes'] = int(nc)

    if not test_fname:
        test_fname = config['DS_TEST_FILE']

    
    ds1 = CnfDataset(config['DS_TRAIN_FILE'],settings['threshold'],mode=settings['data_mode'])
    ds2 = CnfDataset(test_fname, settings['threshold'], ref_dataset=ds1, mode=settings['data_mode'])
    ds3 = CnfDataset(config['DS_VALIDATION_FILE'], settings['threshold'], ref_dataset=ds1, mode=settings['data_mode'])
    settings.hyperparameters['max_clauses'] = ds1.max_clauses
    settings.hyperparameters['max_variables'] = ds1.max_variables
    # net = load_model_from_file()
    # net.load_state_dict(torch.load(model_fname))
    net = torch.load(model_fname)
    ipdb.set_trace()

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
        return rc['config']['hyperparams'], dmode, rc['config']


