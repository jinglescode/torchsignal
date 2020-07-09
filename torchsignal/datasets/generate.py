import numpy as np
import matplotlib.pyplot as plt


def generate_signal(length_seconds, sampling_rate, frequencies_list, add_noise=0, plot=True):
    npnts = sampling_rate*length_seconds  # number of time samples
    time = np.arange(0, npnts)/sampling_rate

    signal = np.zeros(len(time))

    # loop over frequencies to create signal
    for fi in range(0, len(frequencies_list)):
        signal = signal + np.sin(2*np.pi*frequencies_list[fi]*time)

    if add_noise:
        signal = signal + np.random.randn(len(signal))

    if plot:
        plt.plot(time, signal, label='Signal')
        plt.show()

    return signal
