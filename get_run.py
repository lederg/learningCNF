from pymongo import MongoClient
import argparse
import re
import sys
import ipdb
from pprint import pprint

from dispatch_utils import *

def print_experiment(name, hostname, dbname):
    with MongoClient(host=hostname) as client:
        db = client[dbname]
        runs = db['runs']
        k = re.compile(name)
        rc = runs.find({'experiment.name': k})
        for x in rc:
            pprint(x['config'])


def main():
    parser = argparse.ArgumentParser(description='Process some params.')
    parser.add_argument('params', metavar='N', type=str, nargs='*',
                        help='an integer for the accumulator')
    parser.add_argument('--host', type=str, help='Host address') 
    parser.add_argument('-d', '--db', type=str, default='rl_exp', help='Database name')    
    parser.add_argument('-r', '--remote', action='store_true', default=False, help='Use default remote machine (aws01)') 
    args = parser.parse_args()

    assert(len(args.params)>0)
    expname = args.params[0]
    if args.remote:
        hostname = get_mongo_addr('aws01')+':27017'
    elif args.host:
        hostname = args.host
    else:
        hostname = None

    print_experiment(expname,hostname,args.db)
if __name__=='__main__':
    main()
