import os
import numpy as np
import scipy.io as sio
from typing import Tuple

from torchsignal.datasets.dataset import PyTorchDataset


class HSSSVEP(PyTorchDataset):
    """
    This is a private dataset.
    A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces
    Yijun Wang, Xiaogang Chen, Xiaorong Gao, Shangkai Gao
    https://ieeexplore.ieee.org/document/7740878
    Sampling rate: 250 Hz
    """

    def __init__(self, root: str, subject_id: int, verbose: bool = False) -> None:

        self.root = root
        self.sample_rate = 1000
        self.data, self.targets, self.channel_names = _load_data(self.root, subject_id, verbose)

    def __getitem__(self, n: int) -> Tuple[np.ndarray, int]:
        return (self.data[n], self.targets[n])

    def __len__(self) -> int:
        return len(self.data)


def _load_data(root, subject_id, verbose):

    path = os.path.join(root, 'S'+str(subject_id)+'.mat')
    data_mat = sio.loadmat(path)

    raw_data = data_mat['data'].copy()
    raw_data = np.transpose(raw_data, (2,3,0,1))

    data = []
    targets = []
    for target_id in np.arange(raw_data.shape[0]):
        data.extend(raw_data[target_id])
        
        this_target = np.array([target_id]*raw_data.shape[1])
        targets.extend(this_target)

    data = np.array(data)
    targets = np.array(targets)

    channel_names = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','M2','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']

    if verbose:
        print('Load path:', path)
        print('Data shape', data.shape)
        print('Targets shape', targets.shape)

    return data, targets, channel_names
