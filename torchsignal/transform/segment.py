import numpy as np


def segment_signal(signal, window_len, shift_len, sample_rate, add_segment_axis=True, verbose=False):
    r"""
    Divide a signal time domain into length of `window_len`.

    Args:
        signal : ndarray, shape (trial,channel,time)
            Input signal by trials in time domain
        window_len : int
            Window/segment length (in seconds)
        shift_len : int
            Shift of the window (in time samples). Note: indirectly specifies overlap
        sample_rate : int
            Sampling frequency
        add_segment_axis : boolean, default: True
            If True, segmented shape is (trial,channel,#segments,time), otherwise (trial,channel,time)
        verbose : boolean, default: False
            Print details
    Returns:
        segmented : ndarray, shape (trial,channel,#segments,time) or (trial,channel,time)
            Segmented matrix
    Example:
        Fs = 1000 # Sampling frequency
        num_trials = 100
        num_channels = 9
        num_timesamples = 4000
        S = np.zeros((num_trials,num_channels,num_timesamples)) # Signal

        window_len = 1
        shift_len = 1
        segmented = segment_signal(S, window_len, shift_len, Fs, True)
    Dependencies:
        np : numpy package
        buffer : function
    """

    assert len(signal.shape) == 3, "signal shape must be (trial,channel,time)"

    duration = int(window_len*sample_rate)
    # (window_len-shift_len)*sample_rate
    data_overlap = (window_len*sample_rate-shift_len)

    num_segments = int(
        np.ceil((signal.shape[2]-data_overlap)/(duration-data_overlap)))

    if add_segment_axis:  # return (trial,channel,segments,time)
        segmented = np.zeros(
            (signal.shape[0], signal.shape[1], num_segments, duration))
        for trial in range(0, signal.shape[0]):
            for channel in range(0, signal.shape[1]):
                segmented[trial, channel, :, :] = buffer(
                    signal[trial, channel], duration, data_overlap, num_segments)
    else:  # return (trial,channel,time)
        segmented = np.zeros(
            (signal.shape[0]*num_segments, signal.shape[1], duration))
        for trial in range(0, signal.shape[0]):
            for channel in range(0, signal.shape[1]):
                signal_buffer = buffer(
                    signal[trial, channel], duration, data_overlap, num_segments)
                for segment in range(0, signal_buffer.shape[0]):
                    index = (trial*num_segments)+segment
                    segmented[index, channel, :] = signal_buffer[segment]

    if verbose:
        print('Duration', duration)
        print('Overlap', data_overlap)
        print('#segments', num_segments)
        print('Shape from',  signal[0, 0].shape, 'to', buffer(
            signal[0, 0], duration, data_overlap, num_segments).shape)
        print('Input: Signal shape', signal.shape)
        print('Output: Segmented signal shape', segmented.shape)

    return segmented


def buffer(signal, duration, data_overlap, number_segments, verbose=False):
    r"""
    Divide a single signal time domain into length of `duration`.

    Args:
        signal : ndarray, shape (time,)
            Single input signal in time domain
        duration : int
            Window/segment length in time samples
        data_overlap : int
            Segment length that is overlapped in time samples
        number_segments : int
            Number of segments
        verbose : boolean, default: False
            Print details
    Returns:
        segmented : ndarray, shape (#segments,time)
            Segmented signals
    Example:
        Fs = 1000 # Sampling frequency
        L = 4000 # Length of signal
        t = np.arange(0, (L/(Fs)), step=1.0/(Fs))
        S = 0.7*np.sin(2*np.pi*10*t) + np.sin(2*np.pi*12*t) # Signal

        duration = 2000 # in time samples
        data_overlap = 1000 # in time samples
        segmented = buffer(S,duration,data_overlap, True)
    Dependencies:
        np : numpy package
    """

    temp_buf = [signal[i:i+duration]
                for i in range(0, len(signal), (duration-int(data_overlap)))]
    temp_buf[number_segments-1] = np.pad(
        temp_buf[number_segments-1],
        (0, duration-temp_buf[number_segments-1].shape[0]),
        'constant')
    segmented = np.vstack(temp_buf[0:number_segments])
    if verbose:
        print('Input: Signal shape', signal.shape)
        print('Output: Segmented signal shape', segmented.shape)

    return segmented
