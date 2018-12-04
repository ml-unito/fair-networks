.SILENT:

experiment_dirs=$(wildcard experiments/*/)
excluded_experiment_dirs=$(wildcard experiments/_*/)
experiment_list=$(foreach dir, $(filter-out $(excluded_experiment_dirs), $(experiment_dirs)), $(dir)performances.json)


all: representations evaluations
	echo "All Done"

representations: experiments
	echo "Building representations..."
	python code/build_representations.py /app/code experiments
	echo "Building representations: done!"

evaluations: $(experiment_list)
	echo "Evaluations: done!"


clean: clean_evaluations clean_representations
	echo "Cleaning: done!"

clean_evaluations:
	rm -f $(experiment_list)

clean_representations:
	find experiments -name "*_repr.csv" -exec rm {} \;

%performances.json:
	echo "Evaluating performances for experiment $(dir $@)" 
	python code/test_representations.py $(dir $@)
