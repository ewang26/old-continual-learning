#main_batch 

from data import RandomMemorySetManager, KMeansMemorySetManager, \
        LambdaMemorySetManager, GSSMemorySetManager, ClassBalancedReservoirSampling, iCaRL
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


# # Check for M1 Mac MPS (Apple Silicon GPU) support
# if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     print("Using M1 Mac")
#     DEVICE = torch.device("mps")
# # Check for CUDA support (NVIDIA GPU)
# elif torch.cuda.is_available():
#     print("Using CUDA")
#     DEVICE = torch.device("cuda")
# # Default to CPU if neither is available
# else:
#     print("Using CPU")
#     DEVICE = torch.device("cpu")

#DEVICE = torch.device("cpu")

from data import DEVICE


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


import os
import torch
import numpy as np
import multiprocessing

def process_experiment(args):
    p_index, p, sample_num, config = args
    print(f'*** starting experiment for p = {p}')
    print(f'*** RUN = {sample_num}')
    random_seed = int(np.random.default_rng().integers(low=0, high=1e6))
    
    model_save_dir = getattr(config, "model_save_dir", None)
    if model_save_dir is not None and not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model_load_dir = getattr(config, "model_load_dir", None)
    if model_load_dir is not None:
        print("Model load path given so loading model and not training")
        print("If this is unintended behaviour, remove model_load_dir from config")

    if config.memory_set_manager == RandomMemorySetManager:
        memory_set_manager = config.memory_set_manager(p, random_seed=random_seed)
    elif config.memory_set_manager == KMeansMemorySetManager:
        memory_set_manager = config.memory_set_manager(
            p,
            num_centroids=config.num_centroids,
            device=config.device,
            random_seed=random_seed
        )
    elif config.memory_set_manager == LambdaMemorySetManager:
        memory_set_manager = config.memory_set_manager(p)
    elif config.memory_set_manager == GSSMemorySetManager:
        memory_set_manager = config.memory_set_manager(p, random_seed=random_seed)
    elif config.memory_set_manager == ClassBalancedReservoirSampling:
        memory_set_manager = config.memory_set_manager(p, random_seed=random_seed)
    elif config.memory_set_manager == iCaRL:
        memory_set_manager = config.memory_set_manager(p, n_classes=2, random_seed=random_seed)
    else:
        raise ValueError(f"Unsupported memory set manager: {config.memory_set_manager}")

    print(f"Memory Selection Method: {config.memory_set_manager}")

    manager = config.learning_manager(
        memory_set_manager=memory_set_manager,
        use_wandb=config.use_wandb,
        model=config.model,
    )

    num_tasks = manager.num_tasks

    # Train on first task
    final_accs = []
    final_backward_transfers = []

    for task_num in range(num_tasks):
        print(f'*** at task = {task_num}')
        
        if model_load_dir is not None:
            print("EVALUATING")

            for ideal_model_index in range(config.num_ideal_models):
                post_train_model_load_path = (
                    f'{model_load_dir}/ideal_model/train_{ideal_model_index}/task_{task_num}/model.pt'
                )
                post_train_model = torch.load(post_train_model_load_path, map_location=config.device)

                for grad_type in config.grad_type:
                    if not ((grad_type == 'past') and (task_num == 0)): # no past gradients for first task
                        if config.use_random_img:
                            mem_sel_path = f"{model_load_dir}/{config.memory_selection_method}_random_img"
                        else:
                            mem_sel_path = f"{model_load_dir}/{config.memory_selection_method}"
                        if not os.path.exists(mem_sel_path): os.mkdir(mem_sel_path)
                        p_save_path = f"{mem_sel_path}/{p}"
                        if not os.path.exists(p_save_path): os.mkdir(p_save_path)
                        run_save_path = f"{p_save_path}/run_{sample_num}"
                        if not os.path.exists(run_save_path): os.mkdir(run_save_path)
                        specific_run_save_path = f"{run_save_path}/train_{ideal_model_index}"
                        if not os.path.exists(specific_run_save_path): os.mkdir(specific_run_save_path)
                        grad_save_path = f"{specific_run_save_path}/grad_task_{task_num}"
                        if not os.path.exists(grad_save_path): os.mkdir(grad_save_path)
                        loc_save_path = f"{grad_save_path}/{grad_type}_grad"
                        if not os.path.exists(loc_save_path): os.mkdir(loc_save_path)

                        manager.compute_gradients_at_ideal(
                            model=post_train_model,
                            grad_save_path=loc_save_path,
                            p=p,
                            grad_type=grad_type,
                            use_random_img=config.use_random_img
                        )

                    manager.update_memory_set(model=post_train_model, p=p)

        else:
            print("TRAINING")

            if model_save_dir is not None:
                mem_sel_path = f"{model_save_dir}/{config.memory_selection_method}"
                if not os.path.exists(mem_sel_path): os.mkdir(mem_sel_path)
                model_p_save_dir = f'{mem_sel_path}/{p}'
                if not os.path.exists(model_p_save_dir): os.mkdir(model_p_save_dir)
                model_train_save_dir = f'{model_p_save_dir}/train_{sample_num}'
                if not os.path.exists(model_train_save_dir): os.mkdir(model_train_save_dir)
                model_save_path = f"{model_train_save_dir}/task_{task_num}"
                if not os.path.exists(model_save_path):
                    os.mkdir(model_save_path)
            else:
                model_save_path = None

            print(f"Training on Task {task_num}")
            acc, backward_transfer = manager.train(
                epochs=config.epochs,
                batch_size=config.batch_size,
                lr=config.lr,
                use_memory_set=config.use_memory_set,
                model_save_path=model_save_path,
                train_debug=config.train_debug,
                p=p,
                use_weights=True
            )

            final_accs.append(acc)
            final_backward_transfers.append(backward_transfer)

            acc_save_path = model_train_save_dir
            np.save(f'{acc_save_path}/acc.npy', final_accs)
            print(f'acc: {final_accs}')

        if task_num < num_tasks - 1:
            manager.next_task()

    return final_accs, final_backward_transfers

def main(config: Config):
    if config.use_wandb:
        setup_wandb(config)
    
    args_list = []
    rng = np.random.default_rng(seed=config.random_seed)

    for p_index, p in enumerate(config.p_arr):
        num_samples = getattr(config, "num_samples", 0)[p_index]
        for sample_num in range(num_samples):
            args_list.append((p_index, p, sample_num, config))

    pool = multiprocessing.Pool()
    results = pool.map(process_experiment, args_list)
    pool.close()
    pool.join()

    # Aggregate results
    final_accs = []
    final_backward_transfers = []
    for accs, transfers in results:
        final_accs.extend(accs)
        final_backward_transfers.extend(transfers)

    tasks = list(range(len(final_accs)))
    data = [
        [task, final_acc, b_transfer]
        for task, final_acc, b_transfer in zip(tasks, final_accs, final_backward_transfers)
    ]
    table = wandb.Table(
        data=data, columns=["task_idx", "final_test_acc", "final_test_backward_transfer"]
    )

    if config.use_wandb:
        wandb.log({"Metric Table": table})
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
    print("MAIN_BATCH FINISHED")