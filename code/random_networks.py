import sys
import numpy as np
import pandas
from termcolor import colored

sys.path.append('code')

from options import Options

USAGE="""
Proper usage is as follows:
    random_networks <config.fn-json file> --eval-data=<path for the output file> [-H <num features>]

If -H is not given the script will use the number of neurons in the last hidden layer specified
in the .fn-json file.
"""

print(colored("Generating representation using the random model", "green"))
print(colored("Note: this script accept all options of fair_networks.py, but uses only few of them.", "yellow"))
print(colored("\n" + USAGE, "yellow"))


opts = Options()

if(opts.eval_data_path == None):
    print(colored("ERROR: this script can only be used witht the '--eval-data' option", "red"))
    exit(1)

dataset = opts.dataset
num_features = opts.hidden_layers[-1][0]

train_xs, train_ys, train_s = dataset.train_all_data()
model_data_representation = np.random.uniform(size=(len(train_xs), num_features))

result = np.hstack((model_data_representation, train_s, train_ys))
h_header = ["h_"+str(index) for index in range(len(model_data_representation[0]))]
s_header = ["s_"+str(index) for index in range(len(train_s[0]))]
y_header = ["y_"+str(index) for index in range(len(train_ys[0]))]

print(colored("Saving data representations onto %s" % opts.eval_data_path, "green"))
pandas.DataFrame(result, columns=h_header + s_header + y_header).to_csv(opts.eval_data_path, index=False)
