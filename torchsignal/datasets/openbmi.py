import os
import numpy as np
import scipy.io as sio
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset


def _load_data(root, subject_id, session, verbose):

    path = os.path.join(root, 'session'+str(session),
                        's'+str(subject_id)+'/EEG_SSVEP.mat')

    data_mat = sio.loadmat(path)

    objects_in_mat = []
    for i in data_mat['EEG_SSVEP_train'][0][0]:
        objects_in_mat.append(i)

    # data
    data = objects_in_mat[0][:, :, :].copy()
    data = np.transpose(data, (1, 2, 0))
    data = data.astype(np.float32)

    # label
    targets = []
    for i in range(data.shape[0]):
        targets.append([objects_in_mat[2][0][i], 0, objects_in_mat[4][0][i]])
    targets = np.array(targets)
    targets = targets[:, 2]
    targets = targets-1

    # channel
    channel_names = [v[0] for v in objects_in_mat[8][0]]

    if verbose:
        print('Load path:', path)
        print('Objects in .mat', len(objects_in_mat),
              data_mat['EEG_SSVEP_train'].dtype.descr)
        print()
        print('Data shape', data.shape)
        print('Targets shape', targets.shape)

    return data, targets, channel_names


class OPENBMI(Dataset):
    """
    This is a private dataset.
    EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy.
    Min-Ho Lee, O-Yeon Kwon, Yong-Jeong Kim, Hong-Kyung Kim, Young-Eun Lee, John Williamson, Siamac Fazli, Seong-Whan Lee.
    https://academic.oup.com/gigascience/article/8/5/giz002/5304369
    Target frequencies: 5.45, 6.67, 8.57, 12 Hz
    Sampling rate: 1000 Hz
    """

    def __init__(self, root: str, subject_id: int, session: int, verbose: bool = False) -> None:

        self.data, self.targets, self.channel_names = _load_data(
            root, subject_id, session, verbose)
        self.sample_rate = 1000

    def __getitem__(self, n: int) -> Tuple[np.ndarray, int]:
        return (self.data[n], self.targets[n])

    def __len__(self) -> int:
        return len(self.data)

    def set_data_targets(self, data: [] = None, targets: [] = None) -> None:
        if data is not None:
            self.data = data
        if targets is not None:
            self.targets = targets
        