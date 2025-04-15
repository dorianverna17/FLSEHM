#!/bin/bash

#######################
# run enhanced models #
#######################

# list configs in models_config

ls Evaluation/output | mkdir Evaluation/output
ls Evaluation/output/enhanced_model | mkdir Evaluation/output/enhanced_model

cd Flower/FedAvg/Models/models_config/
CONFIG_LIST=()
for config in *.json
do
	CONFIG_LIST+=($config)
done
cd ../../../..

echo $CONFIG_LIST

for config in $CONFIG_LIST
do
	tail_config="${config#/*/*/}"
	file_config="${tail_config%.json}"
	echo Running model enhanced model with $file_config
	ls Evaluation/output/enhanced_model/$file_config | mkdir Evaluation/output/enhanced_model/$file_config
	for i in {0..9}
	do
		echo Iteration $i of model enhanced model run
		touch Evaluation/output/enhanced_model/$file_config/$i.log
		touch Evaluation/output/enhanced_model/$file_config/$i.err
		source Flower/FedAvg/start_simulation.sh enhanced_model Flower/FedAvg/Models/models_config/$config \
			>Evaluation/output/enhanced_model/$file_config/$i.log \
			2>Evaluation/output/enhanced_model/$file_config/$i.err
	done
done
