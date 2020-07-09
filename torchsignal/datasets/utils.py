import numpy as np


def onehot_targets(targets, num_class=4):
    onehot_y = np.zeros((targets.shape[0], num_class))
    onehot_y[np.arange(onehot_y.shape[0]), targets] = 1
    onehot_y = onehot_y.astype(np.long)
    return onehot_y
