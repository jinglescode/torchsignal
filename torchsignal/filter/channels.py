import mne
import numpy as np


def pick_channels(data: np.ndarray,
                  channel_names: [str],
                  selected_channels: [str],
                  verbose: bool = False) -> np.ndarray:

    picked_ch = mne.pick_channels(channel_names, selected_channels)
    data = data[:,  picked_ch, :]

    if verbose:
        print('picking channels: channel_names',
              len(channel_names), channel_names)
        print('picked_ch', picked_ch)
        print()

    del picked_ch

    return data
