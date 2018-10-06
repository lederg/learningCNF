from pymongo import MongoClient
import argparse
import re
import sys
import ipdb
from pprint import pprint
from config import *
from settings import *
from dispatch_utils import *

MONGO_MACHINE = 'russell'

def print_experiment(name, hostname, dbname):
    settings = CnfSettings(cfg())
    with MongoClient(host=hostname) as client:
        db = client[dbname]
        runs = db['runs']
        k = re.compile(name)
        rc = runs.find({'experiment.name': k})
        for x in rc:
            fc = {}
            trivial = ['base_mode', 'seed', 'exp_time']
            print('Experiment: {}'.format(x['experiment']['name']))
            c = x['config']
            for k in c.keys():
                if k not in trivial and (k not in settings.hyperparameters.keys() or settings[k] != c[k]):
                    fc[k] = c[k]
            pprint(fc)


def main():
    parser = argparse.ArgumentParser(description='Process some params.')
    parser.add_argument('params', metavar='N', type=str, nargs='*',
                        help='an integer for the accumulator')
    parser.add_argument('--host', type=str, help='Host address') 
    parser.add_argument('-d', '--db', type=str, default='rl_exp', help='Database name')    
    parser.add_argument('-r', '--remote', action='store_true', default=False, help='Use default remote machine ({})'.format(MONGO_MACHINE)) 
    args = parser.parse_args()

    assert(len(args.params)>0)
    expname = args.params[0]
    if args.remote:
        hostname = get_mongo_addr(MONGO_MACHINE)+':27017'
    elif args.host:
        hostname = args.host
    else:
        hostname = None

    print_experiment(expname,hostname,args.db)
if __name__=='__main__':
    main()
