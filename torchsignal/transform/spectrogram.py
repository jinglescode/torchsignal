from typing import Callable, Optional
import torch
from torch import Tensor


class Spectrogram(torch.nn.Module):
    r"""Create a spectrogram from a audio signal.
    Code taken from TORCHAUDIO.TRANSFORMS[https://pytorch.org/audio/stable/transforms.html]
    TORCH.STFT [https://pytorch.org/docs/stable/generated/torch.stft.html] only accepts input either a 1-D time sequence or a 2-D batch of time sequences.
    
    Parameters:
        waveform (Tensor) – Tensor of audio of dimension (…, time).
    
    Returns:
        Dimension (…, freq, time), where freq is n_fft // 2 + 1 where n_fft is the number of Fourier bins, and time is the number of window hops (n_frame).
    
    Return type:
        Tensor
        
    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float or None, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
    """
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: Optional[float] = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None) -> None:
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.power = power
        self.normalized = normalized

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        return self.spectrogram(waveform, self.pad, self.window, self.n_fft, self.hop_length,
                             self.win_length, self.power, self.normalized)

    def spectrogram(
        self,
        waveform: Tensor,
        pad: int,
        window: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        power: Optional[float],
        normalized: bool
    ) -> Tensor:
        r"""Create a spectrogram or a batch of spectrograms from a raw audio signal.
        The spectrogram can be either magnitude-only or complex.
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time)
            pad (int): Two sided padding of signal
            window (Tensor): Window tensor that is applied/multiplied to each frame/window
            n_fft (int): Size of FFT
            hop_length (int): Length of hop between STFT windows
            win_length (int): Window size
            power (float or None): Exponent for the magnitude spectrogram,
                (must be > 0) e.g., 1 for energy, 2 for power, etc.
                If None, then the complex spectrum is returned instead.
            normalized (bool): Whether to normalize by magnitude after stft
        Returns:
            Tensor: Dimension (..., freq, time), freq is
            ``n_fft // 2 + 1`` and ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """

        if pad > 0:
            # TODO add "with torch.no_grad():" back when JIT supports it
            waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

        # pack batch
        shape = waveform.size()
        waveform = waveform.reshape(-1, shape[-1])

        # default values are consistent with librosa.core.spectrum._spectrogram
        spec_f = torch.stft(
            waveform, n_fft, hop_length, win_length, window, True, "reflect", False, True
        )

        # unpack batch
        spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-3:])

        if normalized:
            spec_f /= window.pow(2.).sum().sqrt()
        if power is not None:
            spec_f = self.complex_norm(spec_f, power=power)

        return spec_f
    
    def complex_norm(
        self,
        complex_tensor: Tensor,
        power: float = 1.0
    ) -> Tensor:
        r"""Compute the norm of complex tensor input.
        Args:
            complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
            power (float): Power of the norm. (Default: `1.0`).
        Returns:
            Tensor: Power of the normed input tensor. Shape of `(..., )`
        """

        # Replace by torch.norm once issue is fixed
        # https://github.com/pytorch/pytorch/issues/34279
        return complex_tensor.pow(2.).sum(-1).pow(0.5 * power)