from pymongo import MongoClient
import getopt
import re
import sys
from pprint import pprint

def print_experiment(name):
    with MongoClient() as client:
        db = client['reinforce']
        # db = client['graph_exp']
        runs = db['runs']
        k = re.compile(name)
        rc = runs.find({'experiment.name': k})
        for x in rc:
            pprint(x['config'])


def main():
    optlist, args = getopt.gnu_getopt(sys.argv[1:], '')
    opts = dict(optlist)
    print_experiment(args[0])


if __name__=='__main__':
    main()
