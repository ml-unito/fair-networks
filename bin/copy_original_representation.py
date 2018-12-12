#!/usr/bin/python

import sys
import numpy as np
import pandas
from termcolor import colored
import logging


from fair.utils.options import Options

RANDOM_SEED = 42

USAGE = """
Proper usage is as follows:
    copy_original_representation <config.json file> --eval-data=<path for the output file>
"""

print(colored("Copying original representation into place", "green"))
print(colored("Note: this script accept all options of fair_networks.py, but uses only few of them.", "yellow"))
print(colored("\n" + USAGE, "yellow"))

logging.basicConfig(level=logging.DEBUG)

opts = Options(sys.argv)

if(opts.eval_data_path == None):
    print(colored(
        "ERROR: this script can only be used witht the '--eval-data' option", "red"))
    exit(1)

x,y,s = opts.dataset.train_all_data()
cols_x = ["h_{:d}".format(i) for i in range(x.shape[1])]
cols_s = ["s_{:d}".format(i) for i in range(s.shape[1])]
cols_y = ["y_{:d}".format(i) for i in range(y.shape[1])]

data = np.hstack([x,s,y])



pandas.DataFrame(data, columns=[cols_x+cols_s+cols_y]).to_csv(opts.eval_data_path, index=False)

