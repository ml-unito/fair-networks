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

        print(  "{classifier}"
                "\t{cp_s_train:2.5f}\t{rnd_s_train:2.4f}"
                "\t{cp_y_train:2.4f}\t{rnd_y_train:2.4f}"
                "\t{cp_s_test:2.5f}\t{rnd_s_test:2.4f}"
                "\t{cp_y_test:2.4f}\t{rnd_y_test:2.4f}".format(
                classifier=classifier, 
                cp_s_train=cp["s"]["train"], rnd_s_train=rp["s"]["train"],
                cp_y_train=cp["y"]["train"], rnd_y_train=rp["y"]["train"],
                cp_s_test=cp["s"]["test"], rnd_s_test=rp["s"]["test"],
                cp_y_test=cp["y"]["test"], rnd_y_test=rp["y"]["test"]))

