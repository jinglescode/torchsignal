import numpy as np
import matplotlib.pyplot as plt


def generate_signal(length_seconds, sampling_rate, frequencies_list, func="sin", add_noise=0, plot=True):
    r"""
    Generate a `length_seconds` seconds signal at `sampling_rate` sampling rate.
    
    Args:
        length_seconds : int
            Duration of signal in seconds (i.e. `10` for a 10-seconds signal)
        sampling_rate : int
            The sampling rate of the signal.
        frequencies_list : 1 or 2 dimension python list a floats
            An array of floats, where each float is the desired frequencies to generate (i.e. [5, 12, 15] to generate a signal containing a 5-Hz, 12-Hz and 15-Hz)
            2 dimension python list, i.e. [[5, 12, 15],[1]], to generate a signal with 2 signals, where the second channel containing 1-Hz signal
        func : string, default: sin
            The periodic function to generate signal, either `sin` or `cos`
        add_noise : float, default: 0
            Add random noise to the signal, where `0` has no noise
        plot : boolean
            Plot the generated signal
    Returns:
        signal : 1d ndarray
            Generated signal, a numpy array of length `sampling_rate*length_seconds`
    """
    
    frequencies_list = np.array(frequencies_list, dtype=object)
    assert len(frequencies_list.shape) == 1 or len(frequencies_list.shape) == 2, "frequencies_list must be 1d or 2d python list"
    
    expanded = False
    if isinstance(frequencies_list[0], int):
        frequencies_list = np.expand_dims(frequencies_list, axis=0)
        expanded = True
        
    npnts = sampling_rate*length_seconds  # number of time samples
    time = np.arange(0, npnts)/sampling_rate
    signal = np.zeros((frequencies_list.shape[0],npnts))
    
    for channel in range(0,frequencies_list.shape[0]):
        for fi in frequencies_list[channel]:
            if func == "cos":
                signal[channel] = signal[channel] + np.cos(2*np.pi*fi*time)
            else:
                signal[channel] = signal[channel] + np.sin(2*np.pi*fi*time)
    
        # normalize
        max = np.repeat(signal[channel].max()[np.newaxis], npnts)
        min = np.repeat(signal[channel].min()[np.newaxis], npnts)
        signal[channel] = (2*(signal[channel]-min)/(max-min))-1
    
    if add_noise:        
        noise = np.random.uniform(low=0, high=add_noise, size=(frequencies_list.shape[0],npnts))
        signal = signal + noise

    if plot:
        plt.plot(time, signal.T)
        plt.show()
    
    if expanded:
        signal = signal[0]
        
    return signal
