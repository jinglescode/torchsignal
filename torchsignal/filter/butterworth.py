import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfiltfilt, freqz
from torchsignal.transform.fft import fast_fourier_transform


def butter_bandpass(lowcut, highcut, sample_rate, order=4, output='ba'):
    r"""
    Create a Butterworth bandpass filter
    Design an Nth-order digital or analog Butterworth filter and return the filter coefficients.

    Args:
        lowcut : int
            Lower bound filter
        highcut : int
            Upper bound filter
        sample_rate : int
            Sampling frequency
        order : int, default: 4
            Order of the filter
        output : string, default: ba
            Type of output {‘ba’, ‘zpk’, ‘sos’}
    Returns:
        butter : ndarray
            Butterworth filter
    Dependencies:
        butter : scipy.signal.butter
    """
    nyq = sample_rate * 0.5
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandpass', output=output)


def butter_bandpass_filter_signal_1d(signal, lowcut, highcut, sample_rate, order, verbose=False):
    r"""
    Digital filter bandpass zero-phase implementation (filtfilt)
    Apply a digital filter forward and backward to a signal

    Args:
        signal : ndarray, shape (time,)
            Single input signal in time domain
        lowcut : int
            Lower bound filter
        highcut : int
            Upper bound filter
        sample_rate : int
            Sampling frequency
        order : int, default: 4
            Order of the filter
        verbose : boolean, default: False
            Print and plot details
    Returns:
        y : ndarray
            Filter signal
    Dependencies:
        filtfilt : scipy.signal.filtfilt
        butter_bandpass : function
        plt : `matplotlib.pyplot` package
        freqz : scipy.signal.freqz
        fast_fourier_transform : function
    """
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order)
    y = filtfilt(b, a, signal)

    if verbose:
        w, h = freqz(b, a)
        plt.plot((sample_rate * 0.5 / np.pi) * w,
                 abs(h), label="order = %d" % order)
        plt.plot([0, 0.5 * sample_rate], [np.sqrt(0.5), np.sqrt(0.5)],
                 '--', label='sqrt(0.5)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        low = max(0, lowcut-(sample_rate/100))
        high = highcut+(sample_rate/100)
        plt.xlim([low, high])
        plt.ylim([0, 1.2])
        plt.title('Frequency response of filter - lowcut:' +
                  str(lowcut)+', highcut:'+str(highcut))
        plt.show()

        # TIME
        plt.plot(signal, label='Signal')
        plt.title('Signal')
        plt.show()

        plt.plot(y, label='Filtered')
        plt.title('Bandpass filtered')
        plt.show()

        # FREQ
        lower_xlim = lowcut-10 if (lowcut-10) > 0 else 0
        fast_fourier_transform(
            signal, sample_rate, plot=True, plot_xlim=[lower_xlim, highcut+20], plot_label='Signal')
        fast_fourier_transform(
            y, sample_rate, plot=True, plot_xlim=[lower_xlim, highcut+20], plot_label='Filtered')

        plt.xlim([lower_xlim, highcut+20])
        plt.ylim([0, 2])
        plt.legend()
        plt.xlabel('Frequency (Hz)')
        plt.show()

        print('Input: Signal shape', signal.shape)
        print('Output: Signal shape', y.shape)
    return y


def butter_bandpass_filter(signal, lowcut, highcut, sample_rate, order, verbose=False):
    r"""
    Digital filter bandpass zero-phase implementation (filtfilt)
    Apply a digital filter forward and backward to a signal

    Dependencies:
        sosfiltfilt : scipy.signal.sosfiltfilt
        butter_bandpass : function
        fast_fourier_transform : function
        plt : `matplotlib.pyplot` package
    Args:
        signal : ndarray, shape (trial,channel,time)
            Input signal by trials in time domain
        lowcut : int
            Lower bound filter
        highcut : int
            Upper bound filter
        sample_rate : int
            Sampling frequency
        order : int, default: 4
            Order of the filter
        verbose : boolean, default: False
            Print and plot details
    Returns:
        y : ndarray
            Filter signal
    """
    sos = butter_bandpass(lowcut, highcut, sample_rate,
                          order=order, output='sos')
    y = sosfiltfilt(sos, signal, axis=2)

    if verbose:
        tmp_x = signal[0, 0]
        tmp_y = y[0, 0]

        # time domain
        plt.plot(tmp_x, label='signal')
        plt.show()

        plt.plot(tmp_y, label='Filtered')
        plt.show()

        # freq domain
        lower_xlim = lowcut-10 if (lowcut-10) > 0 else 0
        fast_fourier_transform(
            tmp_x, sample_rate, plot=True, plot_xlim=[lower_xlim, highcut+20], plot_label='Signal')
        fast_fourier_transform(
            tmp_y, sample_rate, plot=True, plot_xlim=[lower_xlim, highcut+20], plot_label='Filtered')

        plt.xlim([lower_xlim, highcut+20])
        plt.ylim([0, 2])
        plt.legend()
        plt.xlabel('Frequency (Hz)')
        plt.show()

        print('Input: Signal shape', signal.shape)
        print('Output: Signal shape', y.shape)

    return y
