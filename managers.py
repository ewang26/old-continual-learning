from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Type, Set, Dict
from pathlib import Path

import torch
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from jaxtyping import Float
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_
import os

from data import MemorySetManager
from models import MLP, MNLIST_MLP_ARCH, CifarNet, CIFAR10_ARCH, CIFAR100_ARCH
from training_utils import (
    MNIST_FEATURE_SIZE,
    convert_torch_dataset_to_tensor,
    plot_cifar_image,
)
from tasks import Task

DEBUG = False

# Check for M1 Mac MPS (Apple Silicon GPU) support
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Using M1 Mac")
    DEVICE = torch.device("mps")
# Check for CUDA support (NVIDIA GPU)
elif torch.cuda.is_available():
    print("Using CUDA")
    DEVICE = torch.device("cuda")
# Default to CPU if neither is available
else:
    print("Using CPU")
    DEVICE = torch.device("cpu")


DEVICE = torch.device("cpu")


class ContinualLearningManager(ABC):
    """Class that manages continual learning training.

    For each different set of tasks, a different manager should be made.
    For example, one manager for MnistSplit, and one for CifarSplit.
    As much shared functionality as possibly should be abstracted into this
    base class.
    """

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
    ):
        """
        Args:
            memory_set_manager: The memory set manager to use to optionally create memory set.
            model: Model to be trained
            dataset_path: Path to the directory where the dataset is stored. TODO change this
            use_wandb: Whether to use wandb to log training.
        """
        self.use_wandb = use_wandb

        self.model = model

        train_x, train_y, test_x, test_y = self._load_dataset(dataset_path=dataset_path)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.memory_set_manager = memory_set_manager

        self.tasks = self._init_tasks()  # List of all tasks
        self.label_to_task_idx = dict()

        # Update label_to_task_idx
        for i, task in enumerate(self.tasks):
            for label in task.task_labels:
                assert label not in self.label_to_task_idx
                self.label_to_task_idx[label] = i

        self.num_tasks = len(self.tasks)
        self.task_index = (
            0  # Index of the current task, all tasks <= task_index are active
        )

        # Performance metrics
        self.R_full = torch.ones(self.num_tasks, self.num_tasks) * -1   
        

    @abstractmethod
    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them"""
        pass

    @abstractmethod
    def _load_dataset(
        self,
    ) -> Tuple[
        Float[Tensor, "n f"],
        Float[Tensor, "n 1"],
        Float[Tensor, "m f"],
        Float[Tensor, "m 1"],
    ]:
        """Load full dataset for all tasks"""
        pass


    @torch.no_grad()
    def evaluate_task(
        self,
        test_dataloader: Optional[DataLoader] = None,
        model: Optional[nn.Module] = None,
        use_memory_set: bool = False,
        p = 1
    ) -> Tuple[float, float]:
        """Evaluate models on current task.
        
        Args:
            test_dataloader: Dataloader containing task data. If None 
                then test_dataloader up to and including current task 
                is used through self._get_task_dataloaders.
            model: Model to evaluate. If None then use self.model.
        """

        if model is None:
            model = self.model
        if test_dataloader is None:
            _, test_dataloader = self._get_task_dataloaders(
                use_memory_set=use_memory_set, batch_size=64
            )

        current_labels: List[int] = list(self._get_current_labels())
        model.eval()

        # Record RTj values accuracy of the model on task j after training on task T
        # Want to get RTi and loop over i values from 1 to T
        total_correct = 0
        total_examples = 0
        task_wise_correct = [0] * (self.task_index + 1)
        task_wise_examples = [0] * (self.task_index + 1)

        for batch_x, batch_y in test_dataloader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_x)

            # Only select outputs for current labels
            outputs_splice = outputs[:, current_labels]

            task_idxs = torch.tensor(
                [self.label_to_task_idx[y.item()] for y in batch_y]
            )
            correct = torch.argmax(outputs_splice, dim=1) == batch_y

            for i in range(self.task_index + 1): 
                task_wise_correct[i] += torch.sum(correct[task_idxs == i]).item()
                task_wise_examples[i] += torch.sum(task_idxs == i).item()

            total_correct += (
                (torch.argmax(outputs_splice, dim=1) == batch_y).sum().item()
            )
            total_examples += batch_x.shape[0]

        task_accs = [cor/total for cor, total in zip(task_wise_correct, task_wise_examples)]
        #R_ji means we are on task j and evaluating on task i
        # Let T be the current task
        # R_Tj = task_accs[j]
        T = self.task_index
        backward_transfer = 0
        for i in range(T+1):
            self.R_full[T, i] = task_accs[i]
            R_Ti = self.R_full[T, i].item()
            R_ii = self.R_full[i, i].item()

            assert(R_Ti != -1 and R_ii != -1)
            backward_transfer += R_Ti - R_ii

        backward_transfer /= T+1

        test_acc = total_correct / total_examples
        if self.use_wandb:
            wandb.log(
                {
                    f"test_acc_task_idx_{self.task_index}": test_acc,
                    f"backward_transfer_task_idx_{self.task_index}": backward_transfer,
                }
            )

        model.train()

        return test_acc, backward_transfer
    
    def get_forward_pass_gradients(self, X, y, model, criterion, current_labels):
        '''
        model should be on DEVICE
        X, y should be tensor batches
        '''

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        outputs = model(X)

        # Only select outputs for current labels
        outputs_splice = outputs[:, current_labels]
        #print(outputs_splice.type(), y.type())
        loss = criterion(outputs_splice, y)
        loss.backward()

        grad_list = []
        for name, p in model.named_parameters():
            grad_list.append(p.grad.clone().detach().cpu().numpy().flatten())
        
        return np.concatenate(grad_list)
    
    def compute_gradients_at_ideal(
        self,
        model: Optional[nn.Module] = None,
        grad_save_path: Optional[Path] = None,
        p: float = 1
    ):
        """
        Given an ideal model, loop through different p, and evaluate gradients at each end task parameters 
        """

        # load data and labels
        train_dataloader, _ = self._get_task_dataloaders(
                use_memory_set=True, batch_size=64, full_batch = True
            )
        
        current_labels: List[int] = list(self._get_current_labels())

        #create label weights
        label_weights = np.ones(len(current_labels))
        label_weights[:-1] = 1/p
        label_weights = torch.from_numpy(label_weights).float()

        # set model to eval, define loss fn
        model.train()
        model.to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_weights)

        # loop through batches; should only be 1 batch right now
        for batch_x, batch_y in train_dataloader:

            # i think we need to zero gradients here as well? not sure if param grad also accumulates without optimizer
            for param in model.parameters():
                param.grad = None

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_x)

            # Only select outputs for current labels
            outputs_splice = outputs[:, current_labels]
            #print(outputs_splice.type(), batch_y.type())
            loss = criterion(outputs_splice, batch_y)
            loss.backward()

            # Get gradient norms
            l2_sum = 0

            # Record the sum of the L2 norms.
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        # Compute the L2 norm of the gradient
                        l2_norm = torch.norm(param.grad)
                        l2_sum += l2_norm.item()

        if grad_save_path is not None:
            # For now as models are small just saving entire things
            for name, p in model.named_parameters():
                #print(p.grad.clone().detach().cpu().numpy())
                np.save(f'{grad_save_path}/grad_{name}', p.grad.clone().detach().cpu().numpy())

        # now, we should also update the memory set of the current task for use in next task
        if self.memory_set_manager.__class__.__name__ in ['GSSMemorySetManager']:
            
            # get a 1-batch dataloader
            terminal_train_dataloader = self._get_terminal_task_dataloader()
            #print(valid_labels)

            # reset criterion just in case
            criterion = nn.CrossEntropyLoss(label_weights)

            # now we go through samples 1 by 1
            for batch_x, batch_y in terminal_train_dataloader:
                    
                # need to do GSS update
                # zero param gradients just in case
                for param in model.parameters():
                    param.grad = None

                #print(batch_x.shape, batch_y.shape)
                #print(batch_y[0] in valid_labels)
                #print(5 in valid_labels)
                #assert False

                grad_sample = self.get_forward_pass_gradients(batch_x, batch_y, model, criterion, current_labels)

                
                #need grad_batch, or forward pass of sample from memory set

                # zero param gradients again
                for param in model.parameters():
                    param.grad = None

                # create subset dataset from updated memory set
                mem_set_len = len(self.tasks[self.task_index].memory_x)
                #print(mem_set_len)
                n = np.floor(self.tasks[self.task_index].memory_set_manager.gss_p*mem_set_len).astype(np.uint32)+1
                #print(n)
                indices = torch.randperm(mem_set_len)[:n]
                mem_sample_x, mem_sample_y = self.tasks[self.task_index].memory_x[indices], self.tasks[self.task_index].memory_y[indices].long()
                #print(mem_sample_x.shape, mem_sample_y.shape)
                # forward pass subset, compute gradient
                grad_batch = self.get_forward_pass_gradients(mem_sample_x, mem_sample_y, model, criterion, current_labels) if not (mem_set_len == 0) else np.zeros_like(grad_sample)
                #print(grad_batch)
                
                #assert(False)
                self.tasks[self.task_index].update_memory_set(batch_x, batch_y, grad_sample, grad_batch)


        return None

    def train(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.01,
        use_memory_set: bool = False,
        model_save_path : Optional[Path] = None,
        train_debug: bool = False,
        p: float = 1,
        use_weights: bool = False # whether or not to weight class 
    ) -> Tuple[float, float, Dict[str, float]]:
        """Train on all tasks with index <= self.task_index

        Args:
            epochs: Number of epochs to train for.
            batch_size: Batch size to use for training.
            lr: Learning rate to use for training.
            use_memory_set: True then tasks with index < task_index use memory set,
                otherwise they use the full training set.
            save_model_path: If not None, then save the model to this path.

        Returns:
            Final test accuracy.
        """
        self.model.train()
        self.model.to(DEVICE)

        train_dataloader, test_dataloader = self._get_task_dataloaders(
            use_memory_set, batch_size
        )
        current_labels = list(self._get_current_labels())

        #create label weights
        if use_weights:
            label_weights = np.ones(len(current_labels))
            label_weights[:-1] = 1/p
            label_weights = torch.from_numpy(label_weights).float()
            criterion = nn.CrossEntropyLoss(weight = label_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # Train on batches
        #criterion = nn.CrossEntropyLoss(weight = label_weights)  # CrossEntropyLoss for classification tasks
        optimizer = Adam(self.model.parameters(), lr=lr)

        self.model.train()

        # save gradients and model at beginning of task
        if model_save_path is not None: 
            grad_save_path = f'{model_save_path}/start_grad'
            if not os.path.exists(grad_save_path): os.mkdir(grad_save_path)
            torch.save(self.model, f"{grad_save_path}/model.pt") # save model params at start
            self.compute_gradients_at_ideal(self.model, grad_save_path = grad_save_path, p = p)

        callbacks = {'loss': []}
        for _ in tqdm(range(epochs)):
            total_loss = 0
            for batch_x, batch_y in train_dataloader:

                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                optimizer.zero_grad()
                # Forward pass
                outputs = self.model(batch_x)

                outputs = outputs[
                    :, current_labels
                ]  # Only select outputs for current labels
                loss = criterion(outputs, batch_y)
                total_loss += loss.detach().cpu()
                # Backward pass and optimize
                loss.backward()
                # Get gradient norms
                l2_sum = 0

                # Record the sum of the L2 norms.
                with torch.no_grad():
                    count = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            # Compute the L2 norm of the gradient
                            l2_norm = torch.norm(param.grad)
                            l2_sum += l2_norm.item()
                            count += 1

                optimizer.step()

                if self.use_wandb:
                    wandb.log(
                        {
                            f"loss_task_idx_{self.task_index}": loss.item(),
                            f"grad_norm_task_idx_{self.task_index}": l2_sum,
                        }
                    )

                # for the current batch, pass every item through GSS
                '''
                TODO loop through current batch, for each sample get grad of sample + grad of random subset of memory set
                pass values to update memory set of CURRENT task (to be used in later tasks)
                '''
                grad_sample = 0
                grad_batch = 0
                self.tasks[self.task_index].update_memory_set(grad_sample, grad_batch)

            callbacks['loss'].append(total_loss)

            # evaluate model 
            test_acc, test_backward_transfer = self.evaluate_task(test_dataloader, p = p)

        if train_debug:
            for name, p in self.model.named_parameters():
                print(name, p.grad.clone().detach().cpu().numpy())
        
                input()

        # save gradients and model at end of task
        if model_save_path is not None: 
            grad_save_path = f'{model_save_path}/end_grad'
            if not os.path.exists(grad_save_path): os.mkdir(grad_save_path)
            torch.save(self.model, f"{grad_save_path}/model.pt") # save model params at start
            self.compute_gradients_at_ideal(self.model, grad_save_path = grad_save_path, p = p)

        # save training loss
        if model_save_path is not None:
            # For now as models are small just saving entire things
            #torch.save(self.model, f"{model_save_path}/model.pt")
            # for name, p in self.model.named_parameters():
            #     #print(p.grad.clone().detach().cpu().numpy())
            #     np.save(f'{model_save_path}/grad_{name}', p.grad.clone().detach().cpu().numpy())

            # save loss
            np.save(f'{model_save_path}/loss.npy', np.array(callbacks['loss']))

        return test_acc, test_backward_transfer 
    
    def train_new_loss(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.01,
        use_memory_set: bool = False,
        model_save_path : Optional[Path] = None,
        train_debug: bool = False
    ) -> Tuple[float, float, Dict[str, float]]:
        """Train on all tasks with index <= self.task_index

        Args:
            epochs: Number of epochs to train for.
            batch_size: Batch size to use for training.
            lr: Learning rate to use for training.
            use_memory_set: True then tasks with index < task_index use memory set,
                otherwise they use the full training set.
            save_model_path: If not None, then save the model to this path.

        Returns:
            Final test accuracy.
        """
        self.model.train()
        self.model.to(DEVICE)

        train_dataloader_list, test_dataloader, _, _ = self._get_task_dataloaders_new_loss(use_memory_set, batch_size)
        # current_labels: List[int] = list(self._get_current_labels())
        current_labels, label_weights = self._get_current_labels_new_loss()
        #print(current_labels, label_weights)
        # Train on batches
        #criterion_list = [nn.CrossEntropyLoss(weight = label_weights[i]) for i in range(len(train_dataloader_list))] # CrossEntropyLoss for classification tasks
        criterion_list = [nn.CrossEntropyLoss() for i in range(len(train_dataloader_list))]
        optimizer = Adam(self.model.parameters(), lr=lr)

        num_batches = np.ceil(len(train_dataloader_list[0])/batch_size)
        self.model.train()
        for _ in tqdm(range(epochs)):   # basic training loop over multiple iterations/epochs
            iter_dl_list = [iter(dl) for dl in train_dataloader_list]
            for _ in range(int(num_batches)):   # looping over each batch of the data
                batches = [next(idl) for idl in iter_dl_list]
                loss = 0
                optimizer.zero_grad()
                for i in range(len(batches)):   # within each batch, looping over the 
                    batch_x = batches[i][0].to(DEVICE)
                    batch_y = batches[i][1].to(DEVICE)

                    # Forward pass
                    outputs = self.model(batch_x)

                    outputs = outputs[
                       :, current_labels[i]
                    ]  # Only select outputs for current labels
                    #print(outputs, batch_y)
                    loss += criterion_list[i](outputs, batch_y - 2*i)   # TODO: get rid of 2*i to get crossentropy labels to [0,1] in a hacky way
                loss.backward()

                # Get gradient norms
                l2_sum = 0

                # Record the sum of the L2 norms.
                with torch.no_grad():
                    count = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            # Compute the L2 norm of the gradient
                            l2_norm = torch.norm(param.grad)
                            l2_sum += l2_norm.item()
                            count += 1
                optimizer.step()

                if self.use_wandb:
                        wandb.log(
                            {
                                f"loss_task_idx_{self.task_index}": loss.item(),
                                f"grad_norm_task_idx_{self.task_index}": l2_sum,
                            }
                        )

            # evaluate model
            test_acc, test_backward_transfer = self.evaluate_task(test_dataloader)

        if model_save_path is not None:
            # For now as models are small just saving entire things
            torch.save(self.model, f"{model_save_path}/model.pt")
            for name, p in self.model.named_parameters():
                #print(p.grad.clone().detach().cpu().numpy())
                np.save(f'{model_save_path}/grad_{name}', p.grad.clone().detach().cpu().numpy())

        return test_acc, test_backward_transfer

        # PRODUCTION BREAK 03/19

    def create_task(
        self,
        target_labels: Set[int],
        memory_set_manager: MemorySetManager,
        active: bool = False,
    ) -> Task:
        """Generate a  task with the given target labels.

        Args:
            target_labels: Set of labels that this task uses.
            memory_set_manager: The memory set manager to use to create memory set.
            active: Whether this task is active or not.
        Returns:
            Task with the given target labels.
        """
        train_index = torch.where(
            torch.tensor([y.item() in target_labels for y in self.train_y])
        )
        test_index = torch.where(
            torch.tensor([y.item() in target_labels for y in self.test_y])
        )

        train_x = self.train_x[train_index]
        train_y = self.train_y[train_index]
        test_x = self.test_x[test_index]
        test_y = self.test_y[test_index]
        task = Task(train_x, train_y, test_x, test_y, target_labels, memory_set_manager)
        task.active = active

        return task

    def _get_task_dataloaders(
        self, use_memory_set: bool, batch_size: int, full_batch: bool = False
    ) -> Tuple[DataLoader, DataLoader]:
        """Collect the datasets of all tasks <= task_index and return it as a dataloader.

        Args:
            use_memory_set: Whether to use the memory set for tasks < task_index.
            batch_size: Batch size to use for training.
        Returns:
            Tuple of train dataloader then test dataloader.
        """

        # Get tasks
        running_tasks = self.tasks[: self.task_index + 1]
        for task in running_tasks:
            assert task.active

        terminal_task = running_tasks[-1]
        memory_tasks = running_tasks[:-1]  # This could be empty

        # Create a dataset for all tasks <= task_index

        if use_memory_set:
            memory_x_attr = "memory_x"
            memory_y_attr = "memory_y"
            terminal_x_attr = "train_x"
            terminal_y_attr = "train_y"
        else:
            memory_x_attr = "train_x"
            memory_y_attr = "train_y"
            terminal_x_attr = "train_x"
            terminal_y_attr = "train_y"

        test_x_attr = "test_x"
        test_y_attr = "test_y"

        combined_train_x = torch.cat(
            [getattr(task, memory_x_attr) for task in memory_tasks]
            + [getattr(terminal_task, terminal_x_attr)]
        )
        combined_train_y = torch.cat(
            [getattr(task, memory_y_attr) for task in memory_tasks]
            + [getattr(terminal_task, terminal_y_attr)]
        )
        combined_test_x = torch.cat(
            [getattr(task, test_x_attr) for task in running_tasks]
        )
        combined_test_y = torch.cat(
            [getattr(task, test_y_attr) for task in running_tasks]
        )

        # Identify the labels for the combined dataset
        # TODO use this later
        combined_labels = set.union(*[task.task_labels for task in running_tasks])

        # Randomize the train dataset
        n = combined_train_x.shape[0]
        perm = torch.randperm(n)
        combined_train_x = combined_train_x[perm]
        combined_train_y = combined_train_y[perm]

        # Put into batches
        train_dataset = TensorDataset(combined_train_x, combined_train_y)
        test_dataset = TensorDataset(combined_test_x, combined_test_y)
        if full_batch:
            train_dataloader = DataLoader(
                train_dataset, batch_size=len(train_dataset), shuffle=True
            )
            test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        else:
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader
    
    def _get_terminal_task_dataloader(self, batch_size: int = 1) -> Tuple[DataLoader, DataLoader]:
        """Collect the datasets of all tasks < task_index and return it as a dataloader.

        Args:
            use_memory_set: Whether to use the memory set for tasks < task_index.
            batch_size: Batch size to use for training.
        Returns:
            Tuple of train dataloader then test dataloader.
        """

        # Get terminal task
        terminal_task = self.tasks[self.task_index]

        # Create a dataset for all tasks < task_index
        terminal_x_attr = "train_x"
        terminal_y_attr = "train_y"

        combined_train_x = torch.cat(
            [getattr(terminal_task, terminal_x_attr)]
        )
        combined_train_y = torch.cat(
            [getattr(terminal_task, terminal_y_attr)]
        )

        # Randomize the train dataset
        n = combined_train_x.shape[0]
        perm = torch.randperm(n)
        combined_train_x = combined_train_x[perm]
        combined_train_y = combined_train_y[perm]

        # Put into batches
        train_dataset = TensorDataset(combined_train_x, combined_train_y)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        return train_dataloader
    
    def _get_task_dataloaders_new_loss(
        self, use_memory_set: bool, batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Collect the datasets of all tasks <= task_index and return it as a dataloader.

        Args:
            use_memory_set: Whether to use the memory set for tasks < task_index.
            batch_size: Batch size to use for training.
        Returns:
            Tuple of train dataloader then test dataloader.
        """

        # Get tasks (task_index for now is 2)
        running_tasks = self.tasks[: self.task_index + 1]
        for task in running_tasks:
            assert task.active

        terminal_task = running_tasks[-1]
        memory_tasks = running_tasks[:-1]  # This could be empty

        # Create a dataset for all tasks <= task_index

        if use_memory_set:
            memory_x_attr = "memory_x"
            memory_y_attr = "memory_y"
            terminal_x_attr = "train_x"
            terminal_y_attr = "train_y"
        else:
            memory_x_attr = "train_x"
            memory_y_attr = "train_y"
            terminal_x_attr = "train_x"
            terminal_y_attr = "train_y"

        test_x_attr = "test_x"
        test_y_attr = "test_y"

        # creating a list of data to use at a given training step
        combined_train_x = [getattr(task, memory_x_attr) for task in memory_tasks]
        combined_train_x.append(getattr(terminal_task, terminal_x_attr))

        combined_train_y = [getattr(task, memory_y_attr) for task in memory_tasks]
        combined_train_y.append(getattr(terminal_task, terminal_y_attr))
        
        # duplicate the memory sets within the train x and y
        repeated_train_x = []
        repeated_train_y = []
        terminal_task_data_len = len(combined_train_x[-1])
        #terminal_task_data_len = min([len(data) for data in combined_train_x])
        memory_task_data_len = len(combined_train_x[0])
        memory_set_ratio = int(np.floor(terminal_task_data_len/memory_task_data_len) + 1)
        #print('term task data len')
        #print(terminal_task_data_len)
        #print(memory_task_data_len, memory_set_ratio)
        for i in range(len(combined_train_x) - 1):
            # duplicating the memory sets until they are as long as the terminal task dataset length
            # memory_task_data_len = len(combined_train_x[i])
            # memory_set_ratio = int(np.floor(terminal_task_data_len/memory_task_data_len) + 1)
            # repeated train is a list of 
            #print(combined_train_x[i].shape)
            #print(np.tile(combined_train_x[i], (memory_set_ratio, 1)).shape)
            #print(np.tile(combined_train_x[i], (memory_set_ratio, 1))[:terminal_task_data_len].shape)

            #print(combined_train_y[i].shape)
            #print(np.tile(combined_train_y[i], memory_set_ratio).shape)
            #print(np.tile(combined_train_y[i], memory_set_ratio)[:terminal_task_data_len].shape)

            repeated_train_x.append(torch.from_numpy(np.tile(combined_train_x[i], (memory_set_ratio, 1))[:terminal_task_data_len]))
            repeated_train_y.append(torch.from_numpy(np.tile(combined_train_y[i], memory_set_ratio)[:terminal_task_data_len]))
        repeated_train_x.append(getattr(terminal_task, terminal_x_attr))
        repeated_train_y.append(getattr(terminal_task, terminal_y_attr))
        
        # memory set version of the tests
        memory_test_x = [getattr(task, memory_x_attr) for task in memory_tasks]
        memory_test_x.append(getattr(terminal_task, terminal_x_attr))

        memory_test_y = [getattr(task, memory_y_attr) for task in memory_tasks]
        memory_test_y.append(getattr(terminal_task, terminal_y_attr))

        # full version of the test data
        full_test_x = [getattr(task, test_x_attr) for task in running_tasks]
        full_test_y = [getattr(task, test_y_attr) for task in running_tasks]

        # copying code from old test dataloader
        combined_test_x = torch.cat(
            [getattr(task, test_x_attr) for task in running_tasks]
        )
        combined_test_y = torch.cat(
            [getattr(task, test_y_attr) for task in running_tasks]
        )
        test_dataset = TensorDataset(combined_test_x, combined_test_y)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


        # creating a list of dataloaders for the train set, the memory test data, and the full test data for up to the current task
        train_dataloader_list = []
        memory_test_dataloader_list = []
        full_test_dataloader_list = []
        for i in range(len(repeated_train_x)):
            #print(repeated_train_x[i].shape)
            #print(repeated_train_y[i].shape)
            train_dataset = TensorDataset(repeated_train_x[i], repeated_train_y[i])
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False
            )   # setting shuffle to false because we duplicated memory sets and we don't want duplicates in same batch
            train_dataloader_list.append(train_dataloader)

            memory_test_dataset = TensorDataset(memory_test_x[i], memory_test_y[i])
            memory_test_dataloader = DataLoader(
                memory_test_dataset, batch_size=batch_size, shuffle=True
            )
            memory_test_dataloader_list.append(memory_test_dataloader)

            full_test_dataset = TensorDataset(full_test_x[i], full_test_y[i])
            full_test_dataloader = DataLoader(
                full_test_dataset, batch_size=batch_size, shuffle=True
            )
            full_test_dataloader_list.append(full_test_dataloader)

        return train_dataloader_list, test_dataloader, memory_test_dataloader_list, full_test_dataloader_list

    def next_task(self) -> None:
        """Iterate to next task"""
        self.task_index += 1
        if self.task_index >= len(self.tasks):
            raise IndexError("No more tasks")
        self.tasks[self.task_index].active = True

        # update memory set buffer size
        if self.tasks[self.task_index].memory_set_manager.__class__.__name__ in ['GSSMemorySetManager']:
            self.tasks[self.task_index].memory_set_manager.memory_set_size += self.tasks[self.task_index].memory_set_manager.memory_set_inc

    def _get_current_labels(self):
        running_tasks = self.tasks[: self.task_index + 1]
        return set.union(*[task.task_labels for task in running_tasks])

    
    def _get_current_labels_new_loss(self):
        """
        Creates a list of task labels corresponding to the tasks that have been run.
        """
        label_list = []
        label_weights = []
        running_tasks = self.tasks[: self.task_index + 1]
        for task in running_tasks:
            running_task_labels = list(task.task_labels)
            label_list.append(running_task_labels)
            weights = np.zeros(10)
            for label in running_task_labels:
                weights[label] = 1/len(running_task_labels)
            label_weights.append(torch.from_numpy(weights).float())
        return label_list, label_weights
    



class Cifar100Manager(ContinualLearningManager, ABC):
    """ABC for Cifar100 Manager. Handles downloading dataset"""

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
        )

    def _load_dataset(
        self, dataset_path: str
    ) -> Tuple[
        Float[Tensor, "n f"],
        Float[Tensor, "n"],
        Float[Tensor, "m f"],
        Float[Tensor, "m"],
    ]:
        """Load full dataset for all tasks

        Args:
            dataset_path: Path to the directory where the dataset is stored.
        Returns:
            Tuple of train_x, train_y, test_x, test_y
        """
        # Define a transform to normalize the data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Download and load the training data
        trainset = torchvision.datasets.CIFAR100(
            root=dataset_path, train=True, download=True, transform=transform
        )

        # Download and load the testing data
        testset = torchvision.datasets.CIFAR100(
            root=dataset_path, train=False, download=True, transform=transform
        )

        train_x, train_y = convert_torch_dataset_to_tensor(trainset, flatten=False)
        test_x, test_y = convert_torch_dataset_to_tensor(testset, flatten=False)

        return train_x, train_y.long(), test_x, test_y.long()


class Cifar100ManagerSplit(Cifar100Manager):
    """Continual learning on the split Cifar100 task.

    This has 5 tasks, each with 2 labels. [[0-19], [20-39], [40-59], [60-79], [80-99]]
    """

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
        )

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for Cifar100"""

        # TODO Make this task init a function of an input config file
        tasks = []
        label_ranges = [set(range(i, i + 20)) for i in range(0, 100, 20)]
        for labels in label_ranges:
            task = self.create_task(labels, self.memory_set_manager, active=False)
            tasks.append(task)

        tasks[0].active = True
        return tasks


class Cifar10Manager(ContinualLearningManager, ABC):
    """ABC for Cifar10 Manager. Handles dataset loading"""

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
        )

    def _load_dataset(
        self, dataset_path: str
    ) -> Tuple[
        Float[Tensor, "n f"],
        Float[Tensor, "n"],
        Float[Tensor, "m f"],
        Float[Tensor, "m"],
    ]:
        """Load full dataset for all tasks

        Args:
            dataset_path: Path to the directory where the dataset is stored.
        """
        # Define a transform to normalize the data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Download and load the training data
        trainset = torchvision.datasets.CIFAR10(
            root=dataset_path, train=True, download=True, transform=transform
        )

        # Download and load the testing data
        testset = torchvision.datasets.CIFAR10(
            root=dataset_path, train=False, download=True, transform=transform
        )

        # Classes in CIFAR-10 for ref ( "plane", "car", "bird", "cat",
        #                  "deer", "dog", "frog", "horse", "ship", "truck",)

        train_x, train_y = convert_torch_dataset_to_tensor(trainset, flatten=False)
        test_x, test_y = convert_torch_dataset_to_tensor(testset, flatten=False)

        return train_x, train_y.long(), test_x, test_y.long()


class Cifar10ManagerSplit(Cifar10Manager):
    """Continual learning on the classic split Cifar10 task.

    This has 5 tasks, each with 2 labels. [[0,1], [2,3], [4,5], [6,7], [8,9]]
    """

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
        )

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for MNIST"""

        # TODO Make this task init a function of an input config file
        tasks = []
        for i in range(5):
            labels = set([2 * i, 2 * i + 1])
            task = self.create_task(labels, self.memory_set_manager, active=False)
            tasks.append(task)

        tasks[0].active = True
        return tasks


class Cifar10Full(Cifar10Manager):
    """
    Cifar10 but 1 task running all labels.
    """

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
        )

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for MNIST"""

        # TODO Make this task init a function of an input config file
        labels = set(range(10))
        task = self.create_task(labels, self.memory_set_manager, active=False)
        task.active = True
        tasks = [task]

        return tasks


class MnistManager(ContinualLearningManager, ABC):
    """ABC for Mnist Manager. Handles loading dataset"""

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
        )

    def _load_dataset(
        self, dataset_path: str
    ) -> Tuple[
        Float[Tensor, "n f"],
        Float[Tensor, "n"],
        Float[Tensor, "m f"],
        Float[Tensor, "m"],
    ]:
        """Load full dataset for all tasks

        Args:
            dataset_path: Path to the directory where the dataset is stored.
        Returns:
            Tuple of train_x, train_y, test_x, test_y
        """
        # Define a transform to normalize the data
        # transform = transforms.Compose(
        #    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        # )
        transform = transforms.Compose([transforms.ToTensor()])

        # Download and load the training data
        trainset = torchvision.datasets.MNIST(
            root=dataset_path, train=True, download=True, transform=transform
        )

        # Download and load the test data
        testset = torchvision.datasets.MNIST(
            root=dataset_path, train=False, download=True, transform=transform
        )

        test_x, test_y = convert_torch_dataset_to_tensor(testset, flatten=True)
        train_x, train_y = convert_torch_dataset_to_tensor(trainset, flatten=True)

        return train_x, train_y.long(), test_x, test_y.long()


class MnistManager2Task(MnistManager):
    """Continual learning with 2 tasks for MNIST, 0-8 and 9."""

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
        )

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for MNIST"""

        # TODO Make this task init a function of an input config file

        # Set up tasks
        # Task 1 should just contain examples in the dataset with labels from 0-8
        labels = set(range(9))
        task_1 = self.create_task(labels, self.memory_set_manager, active=True)

        # Task 2 should contain examples in the dataset with label 9
        task_2 = self.create_task(set([9]), self.memory_set_manager, active=False)

        return [task_1, task_2]


class MnistManagerSplit(MnistManager):
    """Continual learning on the classic split MNIST task.

    This has 5 tasks, each with 2 labels. [[0,1], [2,3], [4,5], [6,7], [8,9]]
    """

    def __init__(
        self,
        memory_set_manager: MemorySetManager,
        model: nn.Module,
        dataset_path: str = "./data",
        use_wandb=True,
    ):
        super().__init__(
            memory_set_manager=memory_set_manager,
            dataset_path=dataset_path,
            use_wandb=use_wandb,
            model=model,
        )

    def _init_tasks(self) -> List[Task]:
        """Initialize all tasks and return a list of them. For now hardcoded for MNIST"""

        # TODO Make this task init a function of an input config file
        tasks = []
        for i in range(5):
            labels = set([2 * i, 2 * i + 1])
            task = self.create_task(labels, self.memory_set_manager, active=False)
            tasks.append(task)

        tasks[0].active = True
        return tasks
