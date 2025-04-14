#!/bin/bash

# This script aims to run each model a couple of times and plot the median accuracies

EXISTENT_MODELS=(
	"linear_regression"
	"nn_model"
	"nonlinear_nn_model"
)

ls Evaluation/output | mkdir Evaluation/output

for model in "${EXISTENT_MODELS[@]}"
do
	echo Running model $model
	ls Evaluation/output/$model | mkdir Evaluation/output/$model
	for i in {0..9}
	do
		echo Iteration $i of model $model run
		touch Evaluation/output/$model/$i.log
		source Flower/FedAvg/start_simulation.sh $model>Evaluation/output/$model/$i.log 2>Evaluation/output/$model/$i.err
	done
done
