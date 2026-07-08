import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class RoboMimicTaskDataset(Dataset):
    def __init__(self, sequences, actions, task_id=0):
        self.sequences = torch.as_tensor(
            np.asarray(sequences, dtype=np.float32), dtype=torch.float32
        )
        self.labels = torch.as_tensor(
            np.asarray(actions, dtype=np.float32), dtype=torch.float32
        )
        self.task_type = "regression"
        self.task_id = task_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
