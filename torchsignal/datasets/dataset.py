from torch.utils.data import Dataset
import numpy as np


class PyTorchDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.data = self.data.astype(np.float32)
        self.targets = targets
        self.channel_names = None

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

    def set_data_targets(self, data: [] = None, targets: [] = None) -> None:
        if data is not None:
            self.data = data.copy()
        if targets is not None:
            self.targets = targets.copy()

    def set_channel_names(self,channel_names):
        self.channel_names = channel_names