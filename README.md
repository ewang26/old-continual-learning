# Continual Learning Starter Code

This repository contains code for running basic continual learning training pipelines. 

## Getting Started

### Environment Setup

Download packages from `requirements.txt` using: 

```
pip install -r requirements.txt
```

### Running a Pipeline

`main.py` shows an example training pipeline. The following command will run the code: 

```
python main.py --config /path/to/config/file
```

### Config Files

The code uses yaml config files. See an example in  `./configs/exampple_config.yaml`. 
We provide an description of the different config fields: 


* `use_wandb`: if True, then results logged to weights and biases.
* `wandb_project_name`: name of wandb project.
* `wandb_profile`: wandb user profile. 
* `memory_set_manager`: describes the way that memory sets of previous tasks are selected.
* `p`: If using memory set manager, the probability of including a point from the 
task dataset in the memory set. TODO make this generic memory set manager prarams.
* `use_memory_set`: If False, then previous tasks use entire datasets. Else the memory 
set manager is used to select memory subsets.
* `learning_manager`: manager for the continual learning training. Essentially managers 
what different tasks the continual learning training consists of.
* `lr`: learning rate.
* `batch_size`: training batch size.
* `random_seed`: random seed.
* `epochs`: number of training epochs per task.
* `model_save_dir`: directory to save trained models to after each task.
* `model_load_dir`: directory to load trained models from. If given, them 
model_save_dir is ignored and no model will actually be trained, only 
pretrained models are loaded from memory.
* `experiment_tag`: tag put on wandb run.
* `experiment_metadata_path`: path to a csv file that the wandb run ID will be written to. 
Useful for pulling the data and visualizing later.
