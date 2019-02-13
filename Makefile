# Usage: make CKPT="-i PATH TO THE CKPT" creates the representations recovering the given model
#
#



.SILENT:
.SECONDARY:


# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))


main_dir=experiments
experiment_dirs=$(wildcard $(main_dir)/*/)
excluded_experiment_dirs=$(wildcard $(main_dir)/_*/)
non_excluded_experiments=$(filter-out $(excluded_experiment_dirs), $(experiment_dirs))
performance_output_files=$(foreach dir, $(non_excluded_experiments), $(dir)performances.json)
performance_tables=$(foreach dir, $(non_excluded_experiments), $(dir)performances.tsv)
experiment_models=$(foreach dir, $(non_excluded_experiments), $(dir)models/model-final.ckpt.index)
# representation_files=$(foreach dir, $(non_excluded_experiments), \
# 	$(dir)representations/original_repr_train.csv \
# 	$(dir)representations/fair_networks_repr_train.csv \
# 	$(dir)representations/random_networks_repr_train.csv)



all: $(performance_tables)
	echo "All Done"

experimentation: $(experiment_models)
	echo "All Done"

# clean: clean_performances clean_representations clean_performance_tables
# 	echo "Cleaning: done!"

clean_dir:
	:$(call check_defined, DIR)
	rm -f $(DIR)/performances.json
	rm -f $(DIR)/representations/*_repr_*.csv
	rm -f $(DIR)/performances.tsv

deep_clean_dir:
	:$(call check_defined, DIR)
	rm -rf $(DIR)/models/*
	rm -rf $(DIR)/logdir/*
	rm -f $(DIR)/performances.json
	rm -f $(DIR)/representations/*_repr_*.csv
	rm -f $(DIR)/performances.tsv

# clean_performances:
# 	rm -f $(performance_output_files)

# clean_representations:
# 	find experiments -name "*_repr.csv" -exec rm {} \;

# clean_performance_tables:
# 	rm -f $(performance_tables)

%models/model-final.ckpt.index: %config.json 
	git rev-parse HEAD > $(dir $<)/commit-id
	fair_networks $< -B

%representations/fair_networks_repr_train.csv: %config.json
	fair_networks $< -E representations/fair_networks_repr $(CKPT)

%representations/random_networks_repr_train.csv: %config.json
	random_networks $< -E representations/random_networks_repr

%representations/original_repr_train.csv: %config.json
	copy_original_representation $< -E representations/original_repr

%performances.json: %representations/fair_networks_repr_train.csv %representations/random_networks_repr_train.csv %representations/original_repr_train.csv
	echo "Evaluating performances for experiment $(dir $@)" 
	test_representations $(dir $@)

%performances.tsv: %performances.json
	echo "Creating table with results for experiment $@"
	process_performances $(dir $<)/config.json > $@

list_results:
	echo $(foreach result, $(performance_tables), "$(result)\n")

view_results:
	view $(performance_tables)

publish:
	git rev-parse HEAD
	docker build -t gitlab.c3s.unito.it:5000/fresposi/fair-networks .
	docker push gitlab.c3s.unito.it:5000/fresposi/fair-networks

