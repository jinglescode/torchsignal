import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt


def fast_fourier_transform(signal, sample_rate, plot=False, plot_xlim=[0, 80], plot_ylim=None, plot_label=''):
    r"""
    Use Fourier transforms to find the frequency components of a signal buried in noise.
    Reference: https://www.mathworks.com/help/matlab/ref/fft.html
    Args:
        signal : ndarray, shape (time,)
            Single input signal in time domain
        sample_rate: int
            Sampling frequency
        plot : boolean, default: False
            To plot the single-sided amplitude spectrum
        plot_xlim : array of shape [lower, upper], default: [0,80]
            Set a limit on the X-axis between lower and upper bound
        plot_label : string
            a text label for this signal in plot
    Returns:
        P1 : ndarray, shape ((signal_length/2+1),)
            frequency domain
    Example:
        Fs = 1000 # Sampling frequency
        L = 4000 # Length of signal
        t = np.arange(0, (L/(Fs)), step=1.0/(Fs))
        S = 0.7*np.sin(2*np.pi*10*t) + np.sin(2*np.pi*12*t) # Signal
        P = time_to_frequency(S, sample_rate=Fs, signal_length=L, plot=True, plot_xlim=[0,20])
    Dependencies:
        np : numpy package
        plt : matplotlib.pyplot
        fft : scipy.fft.fft
    """

    signal_length = signal.shape[0]

    if signal_length % 2 != 0:
        signal_length = signal_length+1

    y = fft(signal)
    p2 = np.abs(y/signal_length)
    p1 = p2[0:round(signal_length/2+1)]
    p1[1:-1] = 2*p1[1:-1]

    if plot:
        # TODO change to this chart, https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html
        f = sample_rate*np.arange(0, (signal_length/2)+1)/signal_length
        plt.plot(f, p1, label=plot_label)
        plt.xlim(plot_xlim)
        if plot_ylim is not None:
            plt.ylim(plot_ylim)

    return p1
