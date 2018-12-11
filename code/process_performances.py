import json
import sys

RANDOM_NETS_MODEL_NAME = "random_networks_repr.json"

with open(sys.argv[1], "r") as file:
    results = json.load(file)

rnp = results["performances"][RANDOM_NETS_MODEL_NAME] # random networks all performances


for experiment_name in results["performances"]:
    if experiment_name == RANDOM_NETS_MODEL_NAME:
        continue

    print(experiment_name)
    print("    \ttrain\t\t\t\ttest\t\t\t")
    print("clsf\tmod(s)\trnd(s)\tmod(y)\trnd(y)\tmod(s)\trnd(s)\tmod(y)\trnd(y)")

    experiment = results["performances"][experiment_name]

    for classifier in experiment:
        cp = experiment[classifier] # classifier performances
        rp = rnp[classifier] # random networks performances for this classifier

        print("%s\t%2.4f\t%2.4f\t%2.4f\t%2.4f\t%2.4f\t%2.4f\t%2.4f\t%2.4f" % (
            classifier, 
                cp["s"]["train"], rp["s"]["train"], cp["y"]["train"], rp["y"]["train"],
                cp["s"]["test"], rp["s"]["test"], cp["y"]["test"], rp["y"]["test"]
        ))
    


    
