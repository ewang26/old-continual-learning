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

    memory_set_manager = config.memory_set_manager(
        p=config.p, random_seed=config.random_seed
    )

    manager = config.learning_manager(
        memory_set_manager=memory_set_manager,
        use_wandb=config.use_wandb,
        transfer_metrics=config.transfer_metrics,
        model=config.model,
    )

    epochs = config.epochs
    num_tasks = manager.num_tasks

    # Train on first task
    final_accs = []
    final_backward_transfers = []

    final_metrics = {"leep": [], "logme": [], "gbc": []}

    model_save_dir = getattr(config, "model_save_dir", None)
    model_load_dir = getattr(config, "model_load_dir", None)
    if model_load_dir is not None:
        print("Model load path given so loading model and not training")
        print("If this is unintended behaviour, remove model_load_dir from config")

    for i in range(num_tasks):
        metrics = dict()
        if model_load_dir is not None:
            # Load model and run evaluation
            post_train_model_load_path = (
                model_load_dir / f"{config.model_name}_task_{i}.pt"
            )
            post_train_model = torch.load(post_train_model_load_path)
            if i > 0:
                # Can get pre training model and transfer metric value
                pre_train_model_load_path = (
                    model_load_dir / f"{config.model_name}_task_{i-1}.pt"
                )
                pre_train_model = torch.load(pre_train_model_load_path)
                metrics = manager.evaluate_transfer_metrics(model=pre_train_model)
            else:
                metrics = {
                    "leep": None,
                    "logme": None,
                    "gbc": None,
                }  # First task, no transfer metric values

            acc, backward_transfer = manager.evaluate_task(model=post_train_model)
        else:
            # Train model from scratch
            if model_save_dir is not None:
                model_save_path = model_save_dir / f"{config.model_name}_task_{i}.pt"
            else:
                model_save_path = None

            print(f"Training on Task {i}")
            acc, backward_transfer, metrics = manager.train(
                epochs=epochs,
                batch_size=config.batch_size,
                lr=config.lr,
                use_memory_set=config.use_memory_set,
                model_save_path=model_save_path,
            )

        # Collect performance metrics
        final_accs.append(acc)
        final_backward_transfers.append(backward_transfer)
        for metric_name in metrics.keys():
            metric_val = metrics[metric_name]
            final_metrics[metric_name].append(metric_val)

        # Advance the task
        if i < num_tasks - 1:
            manager.next_task()

    # Log all final results
    tasks = list(range(num_tasks))
    data = [
        [task, final_acc, b_transfer, final_leep, final_logme, final_gbc]
        for task, final_acc, b_transfer, final_leep, final_logme, final_gbc in zip_longest(
            tasks,
            final_accs,
            final_backward_transfers,
            final_metrics["leep"],
            final_metrics["logme"],
            final_metrics["gbc"],
        )
    ]
    table = wandb.Table(
        data=data, columns=["task_idx", "final_test_acc", "final_test_backward_transfer", "leep", "logme", "gbc"]
    )  # logme, gbc

    if config.use_wandb:
        wandb.log({"Metric Table": table})

        # Finish wandb run
        wandb.finish()

    # plot_data()


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
