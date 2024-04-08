from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Type, Set

import torch
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from jaxtyping import Float
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset as TorchDataset


class MemorySetManager(ABC):
    @abstractmethod
    def create_memory_set(self, x: Float[Tensor, "n f"], y: Float[Tensor, "n 1"]):
        """Creates a memory set from the given dataset. The memory set is a subset of the dataset."""
        pass


class RandomMemorySetManager(MemorySetManager):
    def __init__(self, p: float, random_seed: int = 42):
        """
        Args:
            p: The probability of an element being in the memory set.
        """
        self.p = p
        self.generator = torch.Generator().manual_seed(random_seed)

    # Randomly select elements from the dataset to be in the memory set.
    def create_memory_set(
        self, x: Float[Tensor, "n f"], y: Float[Tensor, "n"]
    ) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m"]]:
        """Creates random memory set.

        Args:
            x: x data.
            y: y data.
        Return:
            (x_mem, y_mem) tuple.
        """

        memory_set_size = int(x.shape[0] * self.p)
        # Select memory set random elements from x and y, without replacement
        memory_set_indices = torch.randperm(x.shape[0], generator=self.generator)[
            :memory_set_size
        ]
        #print(memory_set_indices)
        memory_x = x[memory_set_indices]
        memory_y = y[memory_set_indices]

        return memory_x, memory_y

class ClassBalancedReservoirSampling:
    def __init__(self, p: float, random_seed: int = 42):
        """
        Initializes the sampling process.

        Args:
            p: Probability of an element being in the memory set.
            random_seed: Seed for the random number generator to ensure reproducibility.
        """
        self.p = p
        self.generator = torch.Generator().manual_seed(random_seed)
        self.memory_x = None
        self.memory_y = None
        self.class_counts_in_memory = {}
        self.stream_class_counts = {}
        self.memory_set_size = 0

    def update_memory_set(self, x_i: torch.Tensor, y_i: torch.Tensor):
        """
        Updates the memory set with the new instance (x_i, y_i), following the reservoir sampling algorithm.

        Args:
            x_i: The instance of x data.
            y_i: The instance of y data (class label).
        """
        y_i_item = y_i.item()
        if len(self.memory_y) < self.memory_set_size:
            # Memory is not filled; simply add the new instance.
            self.memory_x.append(x_i.unsqueeze(0))  # Add batch dimension
            self.memory_y.append(y_i.unsqueeze(0))  # Add batch dimension
            self.class_counts_in_memory[y_i_item] = self.class_counts_in_memory.get(y_i_item, 0) + 1
        else:
            max_class = max(self.class_counts_in_memory, key=self.class_counts_in_memory.get)
            if self.class_counts_in_memory[y_i_item] < self.memory_set_size * self.p:
                # If current class is underrepresented, attempt to replace an instance of the overrepresented class.
                indices_of_max_class = (self.memory_y == max_class).nonzero(as_tuple=True)[0]
                replace_index = indices_of_max_class[torch.randint(len(indices_of_max_class), (1,), generator=self.generator)].item()
                self.memory_x[replace_index] = x_i
                self.memory_y[replace_index] = y_i
                self.class_counts_in_memory[max_class] -= 1
                self.class_counts_in_memory[y_i_item] = self.class_counts_in_memory.get(y_i_item, 0) + 1
            else:
                # Otherwise, decide to replace an instance of the current class based on probability.
                mc = self.class_counts_in_memory[y_i_item]
                nc = self.stream_class_counts[y_i_item]
                if torch.rand(1, generator=self.generator).item() <= mc / nc:
                    indices_of_y_i_class = (self.memory_y == y_i_item).nonzero(as_tuple=True)[0]
                    replace_index = indices_of_y_i_class[torch.randint(len(indices_of_y_i_class), (1,), generator=self.generator)].item()
                    self.memory_x[replace_index] = x_i
                    self.memory_y[replace_index] = y_i

    def create_memory_set(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates the memory set using class-balanced reservoir sampling.

        Args:
            x: Input features as a tensor.
            y: Corresponding labels as a tensor.

        Returns:
            A tuple containing tensors for the memory set's features and labels.
        """
        self.memory_set_size = int(x.shape[0] * self.p)
        self.memory_x, self.memory_y = [], []

        for i in range(x.shape[0]):
            x_i, y_i = x[i], y[i]
            self.stream_class_counts[y_i.item()] = self.stream_class_counts.get(y_i.item(), 0) + 1
            self.update_memory_set(x_i, y_i)

        # Convert lists of tensors to a single tensor for x and y.
        memory_x = torch.cat(self.memory_x, dim=0)
        memory_y = torch.cat(self.memory_y, dim=0)
        
        return memory_x, memory_y

