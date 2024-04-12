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

import numpy as np


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
    def __init__(self, p: float, num_centroids: int, device: torch.device, random_seed: int = 42):
        """
        Args:
            p: The percentage of samples to retain in the memory set.
            num_centroids: The number of centroids to use for K-Means clustering.
            device: The device to use for computations (e.g., torch.device("cuda")).
            random_seed: The random seed for reproducibility.
        """
        self.p = p
        self.num_centroids = num_centroids
        self.device = device
        self.random_seed = random_seed
        
        # Set the random seed for reproducibility
        torch.manual_seed(self.random_seed)
        
        # Initialize dictionaries to store centroids, cluster counters, and memory sets for each class
        self.centroids = {}
        self.cluster_counters = {}
        self.memory_sets = {}
        
    def create_memory_set(self, x: Float[Tensor, "n f"], y: Float[Tensor, "n 1"]) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m 1"]]:
        """Creates memory set using K-Means clustering for each class.
        
        Args:
            x: x data.
            y: y data.
        
        Returns:
            (memory_x, memory_y) tuple, where memory_x and memory_y are tensors.
        """
        n = x.shape[0]
        f = x.shape[1]
        memory_size = int(n * self.p)
        
        # Get unique classes
        classes = torch.unique(y).tolist()
        num_classes = len(classes)
        
        # Calculate the memory size per class
        memory_size_per_class = memory_size // num_classes
        
        # Initialize memory arrays for each class
        memory_x = {}
        memory_y = {}
        memory_distances = {}
        self.memory_set_indices = {}
        
        for class_label in classes:
            memory_x[class_label] = torch.zeros(memory_size_per_class, f).to(self.device)
            memory_y[class_label] = torch.zeros(memory_size_per_class, 1, dtype=torch.long).to(self.device)
            memory_distances[class_label] = torch.full((memory_size_per_class,), float("inf")).to(self.device)
            self.memory_set_indices[class_label] = torch.zeros(memory_size_per_class, dtype=torch.long).to(self.device)
        
        # Iterate over each class
        for class_label in classes:
            # Get samples and labels for the current class
            class_mask = (y == class_label).squeeze()
            class_samples = x[class_mask]
            class_labels = y[class_mask]
            
            # Initialize centroids and cluster counters for the current class if not already initialized
            if class_label not in self.centroids:
                self.centroids[class_label] = torch.randn(self.num_centroids, f).to(self.device)
                self.cluster_counters[class_label] = torch.zeros(self.num_centroids).to(self.device)
            
            # Iterate over the samples of the current class
            for i in range(class_samples.shape[0]):
                sample = class_samples[i].to(self.device)
                label = class_labels[i].item()
                
                # Find the closest centroid for the current class
                distances = torch.sqrt(torch.sum((self.centroids[class_label] - sample) ** 2, dim=1))
                closest_centroid_idx = torch.argmin(distances).item()
                
                # Update the cluster counter and centroid for the current class
                self.cluster_counters[class_label][closest_centroid_idx] += 1
                self.centroids[class_label][closest_centroid_idx] += (sample - self.centroids[class_label][closest_centroid_idx]) / self.cluster_counters[class_label][closest_centroid_idx]
                
                # Update the memory set for the current class
                class_memory_size = memory_x[class_label].shape[0]
                distance = distances[closest_centroid_idx]
                if i < class_memory_size:
                    memory_x[class_label][i] = sample
                    memory_y[class_label][i] = label
                    memory_distances[class_label][i] = distance
                    self.memory_set_indices[class_label][i] = i
                else:
                    max_idx = torch.argmax(memory_distances[class_label])
                    if distance < memory_distances[class_label][max_idx]:
                        memory_x[class_label][max_idx] = sample
                        memory_y[class_label][max_idx] = label
                        memory_distances[class_label][max_idx] = distance
                        self.memory_set_indices[class_label][max_idx] = i
        
        # Concatenate memory sets from all classes
        memory_x_concat = torch.cat(list(memory_x.values()), dim=0)
        memory_y_concat = torch.cat(list(memory_y.values()), dim=0).view(-1)
        
        return memory_x_concat, memory_y_concat
    

# JONATHAN LAMBDA METHOD
class LambdaMemorySetManager(MemorySetManager):
    def __init__(self, p: float, random_seed: int = 42):
        """
        Args:
            p: The probability of an element being in the memory set.
        """
        self.p = p
        self.generator = torch.Generator().manual_seed(random_seed)

    def create_memory_set(self, x: Float[Tensor, "n f"], y: Float[Tensor, "n 1"]):
        # need to take as inputs not only the data (x) and the labels (y), but also the gradients and model architecture
            # perhaps make an if condition for this method in managers.py or use kwargs
        # define T as an array with length n, where n is total number of samples
        # for each data point:
            # calculate r, a forward pass of the point on the network, saving the output layer classification probability for each class (before softmax)
            # calculate R, the outer product of the Kx1 matrix r, following r @ (1-r).T
            # calculate T[i], the trace of the matrix R, which is equivalent to the hessian of the loss wrt the output layer
        # sort T from greatest to least and denote the memory set as the first m elements, where m = n*p.
        # return this memory set.
        pass


# Alan Gradient Sample Selection (GSS)
class GSSMemorySetManager(MemorySetManager):
    def __init__(self, p: float, random_seed: int = 42):
        """
        Args:
            p: fraction of task dataset to be included in replay buffer.
        """
        self.p = p
        self.generator = torch.Generator().manual_seed(random_seed)
        self.gss_p = 0.1

    def create_memory_set(
        self, x: Float[Tensor, "n f"], y: Float[Tensor, "n"]
    ) -> Tuple[Float[Tensor, "m f"], Float[Tensor, "m"]]:
        """Initializes an empty memory replay buffer if training, called when task objects are created
        Else, use ideal model to generate GSS memory set

        Args:
            x: x data.
            y: y data.
        Return:
            (x_mem, y_mem) tuple.
        """
        
        #start out memory buffer with p*task_data_length
        self.memory_set_size = int(x.shape[0] * self.p)
        self.memory_set_inc = self.memory_set_size
        # # Select memeory set random elements from x and y, without replacement
        # memory_set_indices = torch.randperm(x.shape[0], generator=self.generator)[
        #     :self.memory_set_size
        # ]
        # #print(memory_set_indices)
        # memory_x = x[memory_set_indices]
        # memory_y = y[memory_set_indices]

        # return memory_x, memory_y
        return torch.empty(0), torch.empty(0)
    
    def update_GSS_greedy(self, memory_x, memory_y, C_arr, sample_x, sample_y, grad_sample, grad_batch):
        '''
        TODO implement alg 2 in paper here
        memory_x,y = current memory set for the task (to be used in later tasks)
        C_arr = current list of corresponding scores of each element in memory set
        sample_x,y = new sample
        grad_sample = gradient of new sample
        grad_batch = gradent of random batch of memory_x,y
        '''
        
        # first case, if we dont reach maximum size of memory set, just add it
        # if memory_x.shape[0] + 1 >= self.memory_set_size:
        #     print('in gss greedy')
        #     print(memory_x.shape)
        #     print(memory_y.shape)
        #     print(C_arr.shape)

        sample_norm, batch_norm = np.linalg.norm(grad_sample), np.linalg.norm(grad_batch)
        if self.memory_set_size == 0:
            c = 0
        else:
            c = np.dot(grad_sample, grad_batch) / (sample_norm*batch_norm) + 1 if not (sample_norm*batch_norm == 0) else 1 # else dont add it

        if (memory_x.shape[0] < self.memory_set_size):
            memory_x = torch.cat((memory_x, sample_x), 0)
            memory_y = torch.cat((memory_y, sample_y), 0)
            C_arr = np.concatenate((C_arr, np.array([c])), 0)
        else:
            if c < 1:
                P = C_arr / np.sum(C_arr)
                i = np.random.choice(np.arange(self.memory_set_size), p = P)
                r = np.random.rand()
                if r < C_arr[i] / (C_arr[i] + c):
                    memory_x[i] = sample_x
                    memory_y[i] = sample_y
                    C_arr[i] = c
                #     print('replaced!')
                # else:
                #     print('no replace!')

        # if memory_x.shape[0] + 1 >= self.memory_set_size:
        #     print('after update')
        #     print(memory_x.shape)
        #     print(memory_y.shape)
        #     print(C_arr)
        #     input()
        
        return memory_x, memory_y.long(), C_arr