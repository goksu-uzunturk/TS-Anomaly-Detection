import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import utils
from collections import defaultdict
import random

class DatasetCreator(Dataset):
    """
    Base dataset class for anomaly window datasets using traces and future label prediction.
    """
    def __init__(self, traces_windows, traces_labels, num_classes, num_future_samples, stride):
        self.traces_windows = traces_windows
        self.traces_labels = traces_labels
        self.num_classes = num_classes
        self.num_future_samples = num_future_samples
        self.stride = stride

        self.windows = []
        self.label_indices = []
        self.targets = []
        self.augmented_windows = []
        self.future_time_series = []
        self.future_label_indices = []

        self._process_traces()

    def _process_traces(self):
        """
        Prepares training samples: window + one-hot encoded labels + future label sequence.
        """
        if self.num_future_samples % self.stride == 0:
            last_window_removal_count = self.num_future_samples // self.stride
        else:
            last_window_removal_count = self.num_future_samples // self.stride + 1

        for trace_windows, trace_labels in zip(self.traces_windows, self.traces_labels):
            num_windows = len(trace_windows)
            valid_windows = trace_windows[:-last_window_removal_count]
            valid_labels = trace_labels[:-last_window_removal_count]

            self.windows.extend(valid_windows)

            # Augment windows with label one-hot encodings
            for window, window_labels in zip(valid_windows, valid_labels):
                label_one_hot = np.eye(self.num_classes)[window_labels]
                augmented_window = np.concatenate([window, label_one_hot], axis=-1)
                self.augmented_windows.append(augmented_window)

            # Collect future label targets
            for i in range(num_windows - last_window_removal_count):
                future_traces_labels = trace_labels[i + 1 : i + 1 + last_window_removal_count]
                future_labels = []
                for future_window_labels in future_traces_labels:
                    partial_labels = future_window_labels[-self.stride:]
                    future_labels.extend(partial_labels[:self.num_future_samples - len(future_labels)])
                self.targets.append(future_labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.augmented_windows[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.windows)


class BalancedBatchSampler(Sampler):
    """
    Custom sampler to create batches that ensure uniform distribution of specified minority labels.
    """
    def __init__(self, dataset, batch_size, minority_labels=None, replacement=False):
        """
        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): Number of samples per batch.
            minority_labels (list of str): labels to ensure uniform distribution (e.g., ['f', 'g']).
            replacement (bool): Whether to sample with replacement.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.minority_labels = minority_labels if minority_labels is not None else []
        self.replacement = replacement

        # Store indices of samples containing minority labels
        self.minority_indices = {label: [] for label in self.minority_labels}
        self.other_indices = []

        for idx, augmented_window in enumerate(dataset.augmented_windows):
            found_minority = False
            for label in self.minority_labels:
                if (found_minority==False) & (np.any(augmented_window[:, -dataset.num_classes + label])):
                    self.minority_indices[label].append(idx)
                    found_minority = True
                    break  # Avoid adding the same index to multiple label groups
            if not found_minority:
                self.other_indices.append(idx)

        # Shuffle indices
        for label in self.minority_labels:
            random.shuffle(self.minority_indices[label])
        random.shuffle(self.other_indices)

        # Create batches dynamically
        self.balanced_batches = self._create_balanced_batches()

    def _create_balanced_batches(self):
        """
        Creates batches ensuring that each contains at least one sample from each minority label.
        """
        balanced_batches = []
        available_minority_indices = {label: iter(self.minority_indices[label]) for label in self.minority_labels}

        # Generate batches
        while len(self.other_indices) >= self.batch_size:
            batch = []

            # Add one sample from each minority label (if available)
            for label in self.minority_labels:
                try:
                    batch.append(next(available_minority_indices[label]))
                except StopIteration:
                    if self.replacement:
                        for label in self.minority_labels:
                            random.shuffle(self.minority_indices[label])
                        available_minority_indices = {label: iter(self.minority_indices[label]) for label in self.minority_labels}
                        batch.append(next(available_minority_indices[label]))
                    else:
                        pass  # If a label runs out of samples, ignore it

            # Fill the rest of the batch with random samples from other indices
            remaining_samples_needed = self.batch_size - len(batch)
            batch.extend(random.sample(self.other_indices, remaining_samples_needed))
            
            # Remove used indices
            self.other_indices = [idx for idx in self.other_indices if idx not in batch]

            # Shuffle batch for randomness
            random.shuffle(batch)
            balanced_batches.append(batch)

        # Shuffle all batches to prevent ordering bias
        random.shuffle(balanced_batches)
        return balanced_batches

    def __iter__(self):
        """
        Iterates over the dataset in balanced batches.
        """
        for batch in self.balanced_batches:
            yield batch

    def __len__(self):
        """
        Returns the number of batches.
        """
        return len(self.balanced_batches)

