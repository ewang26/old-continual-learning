#data.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Type, Set

import torch
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from jaxtyping import Float
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset as TorchDataset
from sklearn.cluster import KMeans


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
        # Select memeory set random elements from x and y, without replacement
        memory_set_indices = torch.randperm(x.shape[0], generator=self.generator)[
            :memory_set_size
        ]
        #print(memory_set_indices)
        memory_x = x[memory_set_indices]
        memory_y = y[memory_set_indices]

        return memory_x, memory_y

#THEODORA K-MEANS
class KMeansMemorySetManager(MemorySetManager):
    def __init__(self, p: float, num_centroids: int, num_classes: int, device: torch.device, random_seed: int = 42):
        """
        Args:
            p: The percentage of samples to retain in the memory set.
            num_centroids: The number of centroids to use for K-Means clustering.
            num_classes: The number of classes in the dataset.
            device: The device to use for computations (e.g., torch.device("cuda")).
            random_seed: The random seed for reproducibility.
        """
        self.p = p
        self.num_centroids = num_centroids
        self.num_classes = num_classes
        self.device = device
        self.random_seed = random_seed
        
        # Set the random seed for reproducibility
        torch.manual_seed(self.random_seed)
        
    def create_memory_set(self, x: Float[Tensor, "n f"], y: Float[Tensor, "n 1"]) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m 1"]]:
        """Creates memory set using K-Means clustering.
        
        Args:
            x: x data.
            y: y data.
        
        Returns:
            (memory_x, memory_y) tuple, where memory_x and memory_y are tensors.
        """
        n = x.shape[0]
        f = x.shape[1]
        memory_size = int(n * self.p)
        
        # Initialize centroids and cluster counters
        centroids = torch.randn(self.num_centroids, f).to(self.device)
        cluster_counters = torch.zeros(self.num_centroids).to(self.device)
        
        # Initialize memory arrays
        memory_x = torch.zeros(memory_size, f).to(self.device)
        memory_y = torch.zeros(memory_size, 1, dtype=torch.long).to(self.device)
        memory_distances = torch.full((memory_size,), float("inf")).to(self.device)
        self.memory_set_indices = torch.zeros(memory_size, dtype=torch.long).to(self.device)
        
        # Iterate over the dataset
        for i in range(n):
            sample = x[i].to(self.device)
            label = y[i].item()
            
            # Find the closest centroid
            distances = torch.sqrt(torch.sum((centroids - sample) ** 2, dim=1))
            closest_centroid_idx = torch.argmin(distances).item()
            
            # Update the cluster counter and centroid
            cluster_counters[closest_centroid_idx] += 1
            centroids[closest_centroid_idx] += (sample - centroids[closest_centroid_idx]) / cluster_counters[closest_centroid_idx]
            
            # Update the memory set based on the distance and class
            distance = distances[closest_centroid_idx]
            if i < memory_size:
                memory_x[i] = sample
                memory_y[i] = label
                memory_distances[i] = distance
                self.memory_set_indices[i] = i
            else:
                max_idx = torch.argmax(memory_distances)
                if distance < memory_distances[max_idx]:
                    memory_x[max_idx] = sample
                    memory_y[max_idx] = label
                    memory_distances[max_idx] = distance
                    self.memory_set_indices[max_idx] = i
        
        return memory_x, memory_y.view(-1)