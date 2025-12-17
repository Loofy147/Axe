import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticFuturesDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=20, vocab_size=100, n_axes=4, n_paths_per_axis=3):
        self.num_samples, self.seq_len, self.vocab_size = num_samples, seq_len, vocab_size
        self.n_axes, self.n_paths_per_axis = n_axes, n_paths_per_axis
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        data, labels = [], []
        for _ in range(self.num_samples):
            axis_id = np.random.randint(0, self.n_axes * self.n_paths_per_axis)
            # Add some randomness to the sequence, but keep a signal for the axis
            base_sequence = np.arange(self.seq_len) + (axis_id * 10)
            noise = np.random.randint(-5, 5, self.seq_len)
            sequence = (base_sequence + noise) % self.vocab_size
            data.append(sequence.tolist())
            labels.append(axis_id)
        return torch.LongTensor(data), torch.LongTensor(labels)

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]
