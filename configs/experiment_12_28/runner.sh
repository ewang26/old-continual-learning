#! /bin/bash

export WANDB_API_KEY=8877a09c505001125bdfe3f5abe3e9b98c673875


python main.py --config configs/experiment_12_28/config_1.yaml
echo "Finished config 1"

python main.py --config configs/experiment_12_28/config_2.yaml
echo "Finished config 2"

python main.py --config configs/experiment_12_28/config_3.yaml
echo "Finished config 3"

python main.py --config configs/experiment_12_28/config_4.yaml
echo "Finished config 4"

python main.py --config configs/experiment_12_28/config_5.yaml
echo "Finished config 5"

python main.py --config configs/experiment_12_28/config_6.yaml
echo "Finished config 6"

python main.py --config configs/experiment_12_28/config_7.yaml
echo "Finished config 7"

