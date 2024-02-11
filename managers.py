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
                use_memory_set=False, batch_size=64
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

    def train(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.01,
        use_memory_set: bool = False,
        model_save_path : Optional[Path] = None,
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
        current_labels: List[int] = list(self._get_current_labels())
        # Train on batches
        criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification tasks
        optimizer = Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in tqdm(range(epochs)):
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

            # evaluate model 
            test_acc, test_backward_transfer = self.evaluate_task(test_dataloader)

        if model_save_path is not None:
            # For now as models are small just saving entire things
            torch.save(self.model, model_save_path)

        return test_acc, test_backward_transfer 

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
        self, use_memory_set: bool, batch_size: int
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
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def next_task(self) -> None:
        """Iterate to next task"""
        self.task_index += 1
        if self.task_index >= len(self.tasks):
            raise IndexError("No more tasks")
        self.tasks[self.task_index].active = True

    def _get_current_labels(self) -> Set[int]:
        running_tasks = self.tasks[: self.task_index + 1]
        return set.union(*[task.task_labels for task in running_tasks])


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
