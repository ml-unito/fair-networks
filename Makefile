.SILENT:
.SECONDARY:

experiment_dirs=$(wildcard experiments/*/)
excluded_experiment_dirs=$(wildcard experiments/_*/)
non_excluded_experiments=$(filter-out $(excluded_experiment_dirs), $(experiment_dirs))
performance_output_files=$(foreach dir, $(non_excluded_experiments), $(dir)performances.json)
performance_tables=$(foreach dir, $(non_excluded_experiments), $(dir)performances.tsv)
representation_files=$(foreach dir, $(non_excluded_experiments), $(dir)representations/original_repr.csv $(dir)representations/fair_networks_repr.csv $(dir)representations/random_networks_repr.csv)



all: $(performance_tables)
	echo "All Done"

clean: clean_performances clean_representations clean_performance_tables
	echo "Cleaning: done!"

clean_performances:
	rm -f $(performance_output_files)

clean_representations:
	find experiments -name "*_repr.csv" -exec rm {} \;

clean_performance_tables:
	rm -f $(performance_tables)

%representations/fair_networks_repr.csv: %config.json
	fair_networks $< -E representations/fair_networks_repr.csv

%representations/random_networks_repr.csv: %config.json
	random_networks $< -E representations/random_networks_repr.csv

%representations/original_repr.csv: %config.json
	copy_original_representation $< -E representations/original_repr.csv

%performances.json: $(representation_files)
	echo "Evaluating performances for experiment $(dir $@)" 
	test_representations $(dir $@)

%performances.tsv: %performances.json
	echo "Creating table with results for experiment $@"
	process_performances $< > $@
