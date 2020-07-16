import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchsignal.datasets.dataset import PyTorchDataset


def train_test_split(X, y):
    train, test = dataset_split_stratified(
        X, y, k=0, n_splits=4, seed=71, shuffle=True, pytorch_dataset_object=PyTorchDataset)
    return train, test


def dataset_split_stratified(X, y, k=-1, n_splits=3, seed=71, shuffle=True, pytorch_dataset_object=None):
    return_data = []
    skf = StratifiedKFold(
        n_splits=n_splits, random_state=seed, shuffle=shuffle)
    split_data = skf.split(X, y)

    for train_index, test_index in split_data:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if pytorch_dataset_object is not None:
            return_data.append(pytorch_dataset_object(X_train, y_train))
            return_data.append(pytorch_dataset_object(X_test, y_test))
        else:
            return_data.append((X_train, y_train))
            return_data.append((X_test, y_test))

    if k == -1:
        return tuple(return_data)
    else:
        return tuple(return_data)[k*2:k*2+2]


def onehot_targets(targets, num_class=4):
    onehot_y = np.zeros((targets.shape[0], num_class))
    onehot_y[np.arange(onehot_y.shape[0]), targets] = 1
    onehot_y = onehot_y.astype(np.long)
    return onehot_y
