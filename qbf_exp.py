from config import *
from qbf_data import *
from rl_model import *

settings = CnfSettings(cfg())


def main(argv):
	from task_qbf_train import qbf_train_main
	qbf_train_main()


if __name__ == "__main__":
    main(sys.argv)


