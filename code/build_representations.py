import os
import sys
from termcolor import colored
import subprocess
import json


def extract_model_path(config_path):
    config_string = open(config_path, "r").read()
    config = json.loads(config_string)
    return config["checkpoint"]


code_path = sys.argv[1]
experiments_path = sys.argv[2]
approaches = ['fair_networks', 'random_networks']

for dir in os.listdir(experiments_path):
    print(colored("Entering directory %s" % dir, "green"))
    for approach in approaches:
        print(colored("\tBuilding representations with approach %s" % approach, "yellow"))
        try:
            script_path = os.path.join(code_path, approach) + ".py"
            output_path = os.path.join("representations", "%s_repr.csv" % approach)
            config_path = os.path.join(experiments_path, dir, "config.json")

            if os.path.exists(config_path):
                model_path = os.path.join(experiments_path, dir, extract_model_path(config_path))

                # cmd = ["python", script_path, config_path, "-E", output_path, "-c", model_path]
                cmd = ["python", script_path, config_path, "-E", output_path]
                subprocess.check_output(cmd)
            else:
                print(colored("Error: cannot find config file in path %s" % config_path, "red"))

        except subprocess.CalledProcessError as e:
            print(colored("Error in building representation for command %s" % str(cmd), "red"))
            print("Log of the command is:\n %s" % e.output)

print(colored("Done", "green"))
