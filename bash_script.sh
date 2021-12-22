#!/usr/bin/env bash


for dir in ./data/processed/*; do
	abs_path=$(cd "$dir" && pwd)
	echo "$dir"
	python -m src.models.train_model --data_dir "$abs_path/" --nepochs 80
done
