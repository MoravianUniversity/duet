"""
Base class for DUET algorithms.

This includes all code shared between the different DUET algorithms. Much of it is based on the
original CASA495 MATLAB code (https://github.com/yvesx/casa495) and the near-direct port to Python
(https://github.com/BrownsugarZeer/BSS_DUET).

Several improvements have been made to make the program run faster and more efficiently. It runs
about four times faster than the Python code (which itself has a few improvements over the MATLAB
code).

Additionally, it has additions for online processing and multiple channels.

Improvements that could be made:
 * Implement the "big delay" versions from section 8.4 (try both differential and tiling). This
   allows microphones to be placed further apart and still work. One of those two methods is used
   in followup papers. Neither is in the published MATLAB code.
 * Alternative peak finding algorithms such as weighted k-means, model-based peak removal, and
   peak tracking (https://www.researchgate.net/publication/2551938_Real-Time_Time-Frequency_Based_Blind_Source_Separation)
   Also could use something more general like at https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
 * Some of the scaling factors could maybe be eliminated as they likely cancel out or are
   unnecessary. Other values may need to be tuned though (for example, the size of the peak
   prominences to find).
 * General optimizations to reduce array allocations and shared math.
 * All of the parameters of the __init__ method, particularly the window length.
"""

from abc import abstractmethod, ABCMeta
from functools import cached_property

import numpy as np
import scipy as sp
from numpy import ndarray

EPS = np.finfo(np.float64).eps
EPS_F32 = np.finfo(np.float32).eps


class DuetBase(metaclass=ABCMeta):
    """
    Base class for DUET algorithms. This class is not meant to be instantiated
    directly, but rather to be subclassed by specific DUET algorithms.
    """

    @property
    def p(self) -> float:
        """The symmetric attenuation estimator value weights."""
        return self._p

    @property
    def q(self) -> float:
        """The delay estimator value weights."""
        return self._q

    @property
    def stft(self) -> sp.signal.ShortTimeFFT:
        """The ShortTimeFFT object used to compute the STFT of the input signal."""
        return self._stft

    @property
    def sample_rate(self) -> int:
        """The sample rate of the input audio signal in Hz (samples/sec)."""
        return self._stft.fs

    @property
    def window_length(self) -> int:
        """
        The length of the STFT window in samples. Larger values will result in better frequency
        resolution but worse time resolution. Default is 256.

        The original paper uses 1024 for multiple voices, MS paper uses 256 to be more real-time.

        If 1024 with a 16 kHz sampling rate, this would be 64 ms for each time slice.
        If 256 with a 44.1 kHz sampling rate, this would be 5.8 ms for each time slice.
        """
        return self.stft.m_num

    @cached_property
    def frequencies(self) -> ndarray:
        """The frequencies of the STFT, has shape (f,)."""
        # in radians? original is in Hz; the scaling makes them separated by 2pi / window_length
        return self._stft.f[1:] * (2*np.pi/self.sample_rate)

    @cached_property
    def frequencies_f32(self) -> ndarray:
        """The frequencies of the STFT as float32 instead of float64."""
        return self.frequencies.astype(np.float32)


    _force_stereo = True  # change this in subclasses if needed
    _last_audio = None  # last audio signal processed when running online
    _full_audio_len = None  # length of the full audio signal when running online


    def __init__(self, sample_rate: int = 16000, *, window: int|ndarray = 256,
                 p: float = 1.0, q: float = 0.0):
        """
        Initialize the DUET algorithm with the given parameters.

        Arguments
        ---------
        sample_rate : int
            The sample rate of the input audio signal in Hz (samples/sec).
        window : int|ndarray
            The length of the STFT window in samples. Larger values will result in better frequency
            resolution but worse time resolution. Default is 256.
            If an integer is provided, a Hamming window of that length will be used.

            The original paper uses 1024 for multiple voices, MS paper uses 256 to be more
            real-time.

            If 1024 with a 16 kHz sampling rate, this would be 64 ms for each time slice.
            If 256 with a 44.1 kHz sampling rate, this would be 5.8 ms for each time slice.
        p : float
            The symmetric attenuation estimator value weights, default is 1.
        q : float
            The delay estimator value weights, default is 0.
        """
        self._p = p
        self._q = q

        if isinstance(window, int):
            window_length = window
            window = np.hamming(window)
        elif window.ndim != 1:
            raise ValueError("window must be a 1D array or an integer.")
        else:
            window_length = len(window)

        self._stft = sp.signal.ShortTimeFFT(
            window, window_length - window_length // 2,
            sample_rate, phase_shift=None)


    def run(self, x: ndarray) -> ndarray:
        """
        Run the DUET algorithm on the input audio signal. This is the
        "offline" version of the algorithm, which processes the entire input
        signal.

        Arguments
        ---------
        x : ndarray
            The input audio signal, has shape (n_channels, n_samples).

        Returns
        -------
        sources : ndarray
            The separated sources, has shape (n_sources, n_samples).
        """
        x = self._normalize_data(x, self._force_stereo)
        tf, tf_weights, sym_atn, delay = self._compute_all(x)

        sym_atn_peaks, delay_peaks = self._find_peaks(tf_weights, sym_atn, delay)
        if len(sym_atn_peaks) == 0:
            return x

        atn_peaks = self._convert_sym_to_atn(sym_atn_peaks)
        best, sources = self._compute_sources(tf, atn_peaks, delay_peaks)
        sources = self._demix(best, sources)
        return self._convert_to_time_domain(sources, end=x.shape[-1])


    def start(self, x: ndarray) -> ndarray:
        """
        Initialize the online DUET algorithm with the input audio signal. This
        must be called before any call to update(). Calling this again will
        reset the online DUET algorithm and start over.

        Arguments
        ---------
        x : ndarray
            The input audio signal, has shape (n_channels, n_samples). The
            number of samples must be a multiple of the hop length (half the
            window length). This will determine the length of the samples
            returned by all subsequent calls to update().

        Returns
        -------
        sources : ndarray
            The output audio signal. The shape of the array will be
            (n_sources, n_samples) where n_samples is the number of samples in
            the input audio signal.
        """
        #pylint: disable=attribute-defined-outside-init
        x = self._normalize_data(x, self._force_stereo, True)

        # Compute the time-frequency representation of the input audio signal
        # and the symmetric attenuation and delay values.
        # These are saved for the next call to update().
        self._tf = tf = self._construct_spectrogram(x)
        self._sym_atn, self._delay = sym_atn, delay = self._compute_attenuation_and_delay(tf)
        self._tf_weights = tf_weights = self._compute_weights(tf)

        # Save the last `hop`` samples of the input signal for the next call
        self._last_audio = x[:, -self.stft.hop:]
        self._full_audio_len = x.shape[-1]

        # Regular find peaks
        sym_atn_peaks, delay_peaks = self._find_peaks(tf_weights, sym_atn, delay)
        if len(sym_atn_peaks) == 0:
            return x

        # The end is the same as regular
        atn_peaks = self._convert_sym_to_atn(sym_atn_peaks)
        best, sources = self._compute_sources(tf, atn_peaks, delay_peaks)
        sources = self._demix(best, sources)
        return self._convert_to_time_domain(sources, end=x.shape[-1])


    def update(self, x: ndarray, return_full: bool = True) -> ndarray:
        """
        Update the online DUET algorithm with the new audio signal. This must
        be called after start().
        
        Arguments
        ---------
        x : ndarray
            The input audio signal, has shape (n_channels, n_samples). The
            number of samples must be a multiple of the hop length (half the
            window length). The number of channels must be the same as the
            original input audio but the number of samples can be different.
        return_full : bool
            If True, return the full output audio signal, the same length as the
            input audio signal given to start(). If False, return only the most
            recent audio signal. Default is True.

        Returns
        -------
        sources : ndarray
            The output audio signal. The shape of the array will be
            (n_sources, n_samples) where n_samples is the number of samples in
            the input audio signal to start() or the most recent audio signal
            if return_full is False.
        """
        #pylint: disable=attribute-defined-outside-init
        if self._last_audio is None:
            raise ValueError("Call start() first.")

        x = self._normalize_data(x, self._force_stereo, True)
        if x.shape[0] != self._last_audio.shape[0]:
            raise ValueError("The input audio signal must maintain the number of channels.")

        # TODO: if x.shape[1] > initial audio length, then we need to trim it to that length (plus
        # the last audio) and just use start() again?

        # Compute the new time-frequency representation values just on the new audio
        new_tf = self._update_spectrogram(x)
        new_sym_atn, new_delay = self._compute_attenuation_and_delay(new_tf)
        new_tf_weights = self._compute_weights(new_tf)

        # Append the new time-frequency representation values to the existing ones
        self._tf = tf = _roll(self._tf, new_tf)
        self._sym_atn = sym_atn = _roll(self._sym_atn, new_sym_atn)
        self._delay = delay = _roll(self._delay, new_delay)
        self._tf_weights = tf_weights = _roll(self._tf_weights, new_tf_weights)

        # Save the last hop samples of the input signal for the next call
        self._last_audio = x[:, -self.stft.hop:]

        # Update the peaks (subclasses may do an "update" or a full calculation here)
        sym_atn_peaks, delay_peaks = self._find_peaks_update(
            tf_weights, sym_atn, delay, new_tf_weights, new_sym_atn, new_delay
        )
        if len(sym_atn_peaks) == 0:
            return self._convert_to_time_domain(tf, end=self._full_audio_len) if return_full else x

        # The end is the same as regular
        atn_peaks = self._convert_sym_to_atn(sym_atn_peaks)
        best, sources = self._compute_sources(tf, atn_peaks, delay_peaks)
        sources = self._demix(best, sources)
        if return_full:
            return self._convert_to_time_domain(sources, end=self._full_audio_len)
        else:
            start = (sources.shape[-1] - x.shape[-1] // self.stft.hop) * self.stft.hop
            end = start + x.shape[-1]
            return self._convert_to_time_domain(sources, start=start, end=end)


    def reset(self):
        """
        Reset the online DUET algorithm. This will clear all of the internal
        state. This is to release the resources used by the algorithm.
        """
        self._last_audio = None
        del self._tf
        del self._sym_atn
        del self._delay
        del self._tf_weights


    def compute_weights(self, x: ndarray) -> tuple[ndarray, ndarray, ndarray]:
        """
        Compute the weights for every point in the spectrogram along with the
        symmetric attenuation and delay.

        These are the first two and a half steps of the DUET algorithm.

        Arguments
        ---------
        x : ndarray
            The input audio signal, has shape (n_channels, n_samples).

        Returns
        -------
        tf_weights : ndarray
            The weights for every point in the spectrogram, has shape (f, t) or
            (n_channels-1, f, t) if n_channels > 2.
        sym_atn : ndarray
            The symmetric attenuation, has shape (f, t) or (n_channels-1, f, t)
            if n_channels > 2.
        delay : ndarray
            The relative delay, has shape (f, t) or (n_channels-1, f, t) if
            n_channels > 2.
        """
        return self._compute_all(self._normalize_data(x))[1:]


    def _normalize_data(self, x: ndarray, force_stereo: bool = True,
                        require_hop_length: bool = False) -> ndarray:
        """
        Normalize the input audio signal. This makes it in the range of [-1, 1]
        as floats.

        Arguments
        ---------
        x : ndarray
            The input audio signal with at least two channels,
            has shape (n_channels, n_samples)
        force_stereo : bool
            If True, the input audio signal must be stereo (2 channels).
            If False, the input audio signal can have >=2 channels.
        require_hop_length : bool
            If True, the input audio signal must be a multiple of the hop
            length (half the window length).
        
        Returns
        -------
        x : ndarray
            The normalized input audio signal, has shape (n_channels, n_samples)
            and is in the range of [-1, 1] as floats.

        Raises
        ------
        ValueError
            If the input audio signal is not stereo or not of the correct type
        """
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError("The input audio signal must be multichannel.")
        if force_stereo and x.shape[0] != 2:
            raise ValueError("The input audio signal must be stereo.")
        if require_hop_length and x.shape[1] % self.stft.hop != 0:
            raise ValueError("The input audio signal must be a multiple of the hop length.")
        if x.dtype.kind == 'f':
            mx = abs(x).max()
            return x / mx if mx > 1 else x
        if x.dtype.kind == 'i':
            return x / np.iinfo(x.dtype).max
        raise ValueError("The input audio signal must be float or signed int.")


    def _construct_spectrogram(self, x: ndarray) -> ndarray:
        """
        Construct the two-dimensional weighted spectrogram histogram for the two
        channels. Step 1 in the paper.

        This function supports any number of channels.

        Arguments
        ---------
        x : ndarray
            The input audio signal, has shape: (n_channels, n_samples).

        Returns
        -------
        tf : ndarray
            STFT of x without DC component, has shape (n_channels, f, t) with t
            time slices and f frequency bins, complex values.
        """
        tf = self.stft.stft(x)  # 45% of execution time in original
        return tf[:, 1:, :]  # remove DC component (avoid div by zero freq in delay estimation)


    def _update_spectrogram(self, x: ndarray) -> ndarray:
        """
        Update the spectrogram with the new audio signal. This is done by
        adding the previous last time slice to the new audio signal and then
        computing the STFT of the new audio signal.
        """
        # the p0=1 is so that the last hop is used as context but not included in the output
        x = np.concatenate((self._last_audio, x), axis=1)  # type: ignore
        return self.stft.stft(x, p0=1)[:, 1:, :]


    def _compute_attenuation_and_delay(self, tf: ndarray) -> tuple[ndarray, ndarray]:
        """
        Calculate the relative symmetric attenuation (alpha) and delay (delta)
        for each time/frequency point. This gives us phase and amplitude of the
        left and right channels. Step 2 in the paper.

        This function supports any number of channels >= 2 by doing the pairwise
        computation of the attenuation and delay. The results are concatenated
        along the first axis.

        Arguments
        ---------
        tf : ndarray
            The STFT of the input signal, has shape (n_channels, f, t), complex

        Returns
        -------
        alpha : ndarray
            The symmetric attenuation, has shape (f, t) or (n_channels-1, f, t)
            if n_channels > 2.
        delta : ndarray
            The relative delay, has shape (f, t) or (n_channels-1, f, t)
            if n_channels > 2.
        """
        if tf.shape[0] == 2:
            if tf.dtype == np.complex128:
                eps = EPS
                freqs = self.frequencies
            else:
                eps = EPS_F32
                freqs = self.frequencies_f32

            lr_ratio = (tf[1] + eps) / (tf[0] + eps)
            a = np.abs(lr_ratio)
            alpha = a - 1/a
            delta = -np.arctan2(lr_ratio.imag, lr_ratio.real) * (1 / freqs[:, None])
            # TODO: does this allow big delays?
            # do we need to adjust other params to compensate? different filter?
            #sp.ndimage.uniform_filter(delta, 3, mode='wrap', output=delta)
            return alpha, delta
        else:
            # Based on Speech Separation with Microphone Arrays using the Mean Shift Algorithm
            # (Ayllón, Gil-Pita, Manuel Rosa-Zurera, 2012)
            results = []
            for i in range(tf.shape[0]-1):
                results.append(self._compute_attenuation_and_delay(tf[i:i+1, ...]))
            return np.concatenate(results, axis=0)  # type: ignore


    def _compute_weights(self, tf: ndarray) -> ndarray:
        """
        Compute the weights for every point in the spectrogram. First part of
        step 3 in the paper.

        This function supports any number of channels >= 2 by doing the pairwise
        computation of the attenuation and delay. The results are concatenated
        along the first axis.

        Arguments
        ---------
        tf : ndarray
            The STFT of the input signal, has shape (n_channels, f, t), complex
        
        Returns
        -------
        tf_weight : ndarray
            The weights for every point in the spectrogram, has shape (f, t) or
            (n_channels-1, f, t) if n_channels > 2.
        """
        if tf.shape[0] == 2:
            p, q = self.p, self.q
            tf_weight = np.abs(tf[0])*np.abs(tf[1])
            if p != 1.0:
                tf_weight **= p
            if q != 0.0:  # if q == 0, h2 would be all 1s
                tf_weight *= (np.abs(self.frequencies) ** q)[:, None]
            return tf_weight
        else:
            results = []
            for i in range(tf.shape[0]-1):
                results.append(self._compute_weights(tf[i:i+1, ...]))
            return np.concatenate(results, axis=0)


    def _compute_all(self, x: ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Compute the weights for every point in the spectrogram along with the
        symmetric attenuation and delay.

        These are the first two and a half steps of the DUET algorithm.

        Arguments
        ---------
        x : ndarray
            The input audio signal, has shape (n_channels, n_samples).

        Returns
        -------
        tf : ndarray
            The STFT of the input signal, has shape (n_channels, f, t), complex
        tf_weights : ndarray
            The weights for every point in the spectrogram, has shape (f, t) or
            (n_channels-1, f, t) if n_channels > 2.
        sym_atn : ndarray
            The symmetric attenuation, has shape (f, t) or (n_channels-1, f, t)
            if n_channels > 2.
        delay : ndarray
            The relative delay, has shape (f, t) or (n_channels-1, f, t) if
            n_channels > 2.
        """
        tf = self._construct_spectrogram(x)
        sym_atn, delay = self._compute_attenuation_and_delay(tf)
        tf_weights = self._compute_weights(tf)
        return tf, tf_weights, sym_atn, delay


    @abstractmethod
    def _find_peaks(self, tf_weights: ndarray, sym_atn: ndarray, delay: ndarray
                    ) -> tuple[ndarray, ndarray]:
        """
        Find the peaks in the spectrogram. This is done by finding the local
        maxima in the symmetric attenuation and delay.

        The actual implementation of this function is left to the subclasses, as
        different DUET algorithms may have different methods for finding the peaks.

        Arguments
        ---------
        tf_weights : ndarray
            The weights for every point in the spectrogram, has shape (f, t) or
            (n_channels-1, f, t) if n_channels > 2.
        sym_atn : ndarray
            The symmetric attenuation, has shape (f, t) or (n_channels-1, f, t)
            if n_channels > 2.
        delay : ndarray
            The relative delay, has shape (f, t) or (n_channels-1, f, t) if
            n_channels > 2.

        Returns
        -------
        sym_atn_peaks : ndarray
            The peaks of symmetric attenuation, has shape (n_peaks,)
        delay_peaks : ndarray
            The peaks of delay, has shape (n_peaks,)
        """
        raise NotImplementedError("This method must be overridden by subclasses.")


    def _find_peaks_update(self, tf_weights: ndarray, sym_atn: ndarray, delay: ndarray,
                           tf_weights_new: ndarray, sym_atn_new: ndarray, delay_new: ndarray  #pylint: disable=unused-argument
                           ) -> tuple[ndarray, ndarray]:
        """
        Find peaks during an update step. This is given the full variables used
        by _find_peaks() along with just the new values so that the algorithm
        can possibly just use the new data if possible.

        Default implementation just calls _find_peaks() with the full data.

        Arguments
        ---------
        tf_weights : ndarray
            The weights for every point in the spectrogram, has shape (f, t) or
            (n_channels-1, f, t) if n_channels > 2.
        sym_atn : ndarray
            The symmetric attenuation, has shape (f, t) or (n_channels-1, f, t)
            if n_channels > 2.
        delay : ndarray
            The relative delay, has shape (f, t) or (n_channels-1, f, t) if
            n_channels > 2.
        tf_weights_new : ndarray
            The new time-frequency representation weights.
        sym_atn_new : ndarray
            The new symmetric attenuation values.
        delay_new : ndarray
            The new delay values.

        Returns
        -------
        sym_atn_peaks : ndarray
            The peaks of symmetric attenuation, has shape (n_peaks,)
        delay_peaks : ndarray
            The peaks of delay, has shape (n_peaks,)
        """
        return self._find_peaks(tf_weights, sym_atn, delay)


    def _convert_sym_to_atn(self, sym_atn: ndarray) -> ndarray:
        """Convert the symmetric attenuation to attenuation."""
        return 0.5 * (sym_atn + np.sqrt(sym_atn*sym_atn + 4))


    def _compute_sources(self, tf: ndarray, alphas: ndarray, deltas: ndarray) -> \
        tuple[ndarray, ndarray]:
        """
        Determine the spectrograms for each of the sources. Assigns each
        time-frequency frame to the nearest peak in phase/amplitude space and
        then partitions the spectrogram into sources (one peak per source).
        Step 5 and first part of 6 in the paper.

        This function only supports stereo input (2 channels) and assumes the
        left channel is at index 0 and the right channel is at index 1.

        Arguments
        ---------
        tf : ndarray
            The STFT of the input signal, has shape (2, f, t), complex
        alphas : ndarray
            Peaks of attenuation, has shape (n_sources,)
        delay_peak : ndarray
            Peaks of delay, has shape (n_sources,)

        Returns
        -------
        best : ndarray
            The best source for each t/f pair, has shape (f, t), int
        sources : ndarray
            The possible sources, has shape (n_sources, f, t), complex
        """
        # TODO: support >2 channels

        # adjust dimensions for broadcasting (not the best for argmin, better for building masks)
        # dimensions are (n_peaks, f, t)
        alphas = alphas[:, None, None]
        deltas = deltas[:, None, None]
        freqs = self.frequencies[None, :, None] if tf.dtype == np.complex128 else \
            self.frequencies_f32[None, :, None]
        #tf = tf[:, None, :, :]

        # compute the best mask for each peak
        core = alphas * np.exp(-1j*(freqs*deltas))
        denom = 1 / (1 + alphas * alphas)
        # some rules that may speed up the computation:
        #  - cos(-x) == cos(x)
        #  - sin(-x) == -sin(x)
        #  - exp(1j*x) == cos(x) + 1j*sin(x)
        #  - exp(-1j*x) == cos(x) - 1j*sin(x)
        #  - conj(exp(-1j*x)) == exp(1j*x)
        #  - don't need to compute final sqrt and squares since we are using argmin
        scores = np.abs(core * tf[0] - tf[1]) ** 2 * denom
        best = scores.argmin(axis=0)

        # compute the complex-valued FT source for each peak
        sources = (core * tf[1] + tf[0]) * denom  # 3% of execution time in original

        return best, sources


    def _compute_sources_multi(self, tf: ndarray, alphas: ndarray, deltas: ndarray) -> \
        tuple[ndarray, ndarray]:
        """
        Arguments
        ---------
        tf : ndarray
            The STFT of the input signal, has shape (n_channels, f, t), complex
        alphas : ndarray
            Peaks of attenuation, has shape (n_channels-1, n_sources)
        deltas : ndarray
            Peaks of delay, has shape (n_channels-1, n_sources)

        Returns
        -------
        best : ndarray
            The best source for each t/f pair, has shape (f, t), int
        sources : ndarray
            The possible sources, has shape (n_sources, f, t), complex
        """
        # pylint: disable=invalid-name
        # adjust dimensions for broadcasting (not the best for argmin, better for building masks)
        # dimensions are (n_channels-1 [m], n_sources [p], f/k, t/l)
        alphas = alphas[..., None, None]               # (1, 4,   1,   1)   float
        deltas = deltas[..., None, None]               # (1, 4,   1,   1)   float
        freqs = self.frequencies[None, None, :, None]  # (1, 1, 128,   1)   float
        tf = tf[:, None, :, :]                         # (2, 1, 128, 512)   complex
        print(alphas.shape, deltas.shape, freqs.shape, tf.shape)

        A_mp = alphas * np.exp(1j*(freqs*deltas))          # (1, 4, 128,   1)   complex
        # TODO: how to deal with n_channels vs n_channels-1?
        A_mp_extra = np.ones(A_mp.shape, np.complex128)    # (1, 4, 128,   1)   complex  (1+0j)
        A_mp = np.concatenate((A_mp, A_mp_extra), axis=0)  # (2, 4, 128, 512)   complex

        denom = 1 / (np.abs(A_mp) ** 2).sum(axis=0)  # sum over m
        sources = (tf * np.conj(A_mp)).sum(axis=0) * denom
        SML_p = sources[None, :, :, :]  # add back the m dimension

        L_p_part = abs(tf - A_mp * SML_p)
        L_p = (L_p_part * L_p_part).sum(axis=0)  # sum over m
        best = L_p.argmin(axis=0)

        return best, sources


    def _demix(self, best: ndarray, sources: ndarray) -> ndarray:
        """
        Separate each source. End of step 6 in the paper.

        Arguments
        ---------
        best : ndarray
            The best source for each t/f pair, has shape (f, t), int
        sources : ndarray
            The STFT of the input signals, has shape (n_sources, f, t), complex

        Returns
        -------
        sources : ndarray
            The demixxed sources, has shape (n_sources, f+1, t), complex
            The sources have the DC component added back to them.
        """
        n_sources, _, n_times = sources.shape

        # convert source indices to masks
        masks = np.empty(sources.shape, bool)
        for i in range(n_sources):
            masks[i, ...] = best == i

        # compute the complex-valued FT source for each peak
        return np.concatenate((
            np.zeros((n_sources, 1, n_times)),  # add zeros for the DC component
            sources * masks
        ), axis=1)


    def _convert_to_time_domain(self, sources: ndarray,
                                start: int = 0, end: int|None = None) -> ndarray:
        """
        Convert to sources to time domain to rebuild the original audio. Step 7
        in the paper.

        Arguments
        ---------
        sources : ndarray
            The demixxed sources, has shape (n_sources, f, t), complex
        start : int
            The start index of the time domain signal to return. This is used
            to trim the start for online processing. Default is 0.
        end : int|None
            The end index of the time domain signal to return. Default is None,
            which returns all samples.

        Returns
        -------
        est : ndarray
            seperated audio of all sources, has shape (n_sources, n_samples)
        """
        # 24% of execution time in original
        return self.stft.istft(sources, k0=start, k1=end)


def _roll(data: ndarray, new: ndarray) -> ndarray:
    """
    Roll the data to the left by the length of the new data and concatenate
    the new data to the end of the data.    
    """
    return np.concatenate((data[..., new.shape[-1]-1:-1], new), axis=-1)
