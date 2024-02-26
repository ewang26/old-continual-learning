from data import RandomMemorySetManager
from managers import MnistManagerSplit, Cifar10ManagerSplit, Cifar100ManagerSplit
from configs.config import Config
from pathlib import Path
from itertools import zip_longest
import os
import csv

import wandb
import torch
import random
import numpy as np

import yaml
import argparse


def setup_wandb(config: Config):
    run_name = config.run_name
    experiment_tag = getattr(config, "experiment_tag", None)
    experiment_metadata_path = getattr(config, "experiment_metadata_path", None)
    tags = [experiment_tag] if experiment_tag is not None else []

    run = wandb.init(
        tags=tags,
        project=config.wandb_project_name,
        entity=config.wandb_profile,
        name=run_name,
        config=config.config_dict,
    )

    if experiment_metadata_path is not None:
        # Create csv storing run ida
        new_row = [run.path]
        file_exists = os.path.exists(experiment_metadata_path)
        # Open the file in append mode ('a') if it exists, otherwise in write mode ('w')
        with open(
            experiment_metadata_path, mode="a" if file_exists else "w", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(new_row)


def main(config: Config):
    if config.use_wandb:
        setup_wandb(config)

    for p in config.p_arr:
        print(f'*** starting experiment for p = {p}')

        num_samples = getattr(config, "num_samples", 1)

        for sample_num in range(num_samples):
            random_seed = np.random.randint(0,1e6)

            memory_set_manager = config.memory_set_manager(
                p, random_seed=random_seed
            )

            manager = config.learning_manager(
                memory_set_manager=memory_set_manager,
                use_wandb=config.use_wandb,
                model=config.model,
            )

            epochs = config.epochs
            num_tasks = manager.num_tasks

            # Train on first task
            final_accs = []
            final_backward_transfers = []

            model_save_dir = getattr(config, "model_save_dir", None)
            #model_save_dir = f'{model_save_dir}/{p}/'
            model_load_dir = getattr(config, "model_load_dir", None)
            if model_load_dir is not None:
                print("Model load path given so loading model and not training")
                print("If this is unintended behaviour, remove model_load_dir from config")

            for task_num in range(num_tasks):
                if model_load_dir is not None:
                    # Load model and run evaluation
                    post_train_model_load_path = (
                        f'{model_load_dir}/1/task_{task_num}/model.pt'
                    )
                    post_train_model = torch.load(post_train_model_load_path)
                    # Can get pre training model 
                    acc, backward_transfer = manager.evaluate_task(model=post_train_model)

                    # save gradients w.r.t ideal weights
                    p_save_path = f"{model_load_dir}/{p}" # save path for 0.x of memory set
                    if not os.path.exists(p_save_path): os.mkdir(p_save_path)
                    run_save_path = f"{p_save_path}/run_{sample_num}" # save path for a specific run
                    if not os.path.exists(run_save_path): os.mkdir(run_save_path)
                    grad_save_path = f"{run_save_path}/ideal_grad_task_{task_num}"
                    if not os.path.exists(grad_save_path):
                        os.mkdir(grad_save_path)
                    manager.compute_gradients_at_ideal(
                        model = post_train_model,
                        grad_save_path = grad_save_path)
                else:
                    # Train model from scratch
                    if model_save_dir is not None:
                        #create save dir
                        if not os.path.exists(model_save_dir):
                            os.mkdir(model_save_dir)
                        #create task specific save dir
                        model_save_path = f"{model_save_dir}/{p}/task_{task_num}"
                        if not os.path.exists(model_save_path):
                            os.mkdir(model_save_path)
                    else:
                        model_save_path = None

                    print(f"Training on Task {task_num}")
                    acc, backward_transfer = manager.train(
                        epochs=epochs,
                        batch_size=config.batch_size,
                        lr=config.lr,
                        use_memory_set=config.use_memory_set,
                        model_save_path=model_save_path,
                    )

                # Collect performance metrics
                final_accs.append(acc)
                final_backward_transfers.append(backward_transfer)

                # Advance the task
                if task_num < num_tasks - 1:
                    manager.next_task()

        # Log all final results
        tasks = list(range(num_tasks))
        data = [
            [task, final_acc, b_transfer]
            for task, final_acc, b_transfer in zip_longest(
                tasks,
                final_accs,
                final_backward_transfers,
            )
        ]
        table = wandb.Table(
            data=data, columns=["task_idx", "final_test_acc", "final_test_backward_transfer"]
        )  

    if config.use_wandb:
        wandb.log({"Metric Table": table})
        # Finish wandb run
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a maze controller")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration file to run from.",
    )
    args = parser.parse_args()

    with open(f"{args.config}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    config = Config(config_dict)

    main(config)
