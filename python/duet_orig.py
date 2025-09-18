"""
Original DUET (Degenerate Unmixxing Estimation Technique) Algorithm.

Based on the original CASA495 MATLAB code (https://github.com/yvesx/casa495) and the near-direct
port to Python (https://github.com/BrownsugarZeer/BSS_DUET).

The paper can be found at:
https://www.researchgate.net/publication/227143748_The_DUET_blind_source_separation_algorithm

(The very original paper is https://personal.math.ubc.ca/~oyilmaz/preprints/icassp_bss1.pdf
but that was lacking some of the components of the updated paper).

Several improvements have been made to make the program run faster and more efficiently. It runs
about four times faster than the Python code (which itself has a few improvements over the MATLAB
code).

Improvements that could be made:
 * Same as for DUET:
    * "big delay" support
    * removal of scaling factors
    * general optimizations

Should Tune:
 * All of the parameters of the __init__ method, particularly the window length.
"""


import numpy as np
import scipy as sp
from numpy import ndarray
from skimage import morphology

from duet_base import DuetBase


class DuetOrig(DuetBase):
    """
    Original DUET (Degenerate Unmixxing Estimation Technique) algorithm implementation.
    """
    # TODO: several things to think about for bins:
    # - use size of bins instead of number of bins? we can figure out the size from there
    #  (the paper's math uses bin size, but the code uses number of bins and extent)
    #  (the MS version uses bin size as well (actually bandwidth, but similar concept))
    # - many attenuation values are outside of the default max in x.wav
    # - not many delays outside of the default max in x.wav

    @property
    def attenuation_max(self) -> float:
        """The maximum magnitude of symmetric attenuation to consider."""
        return self._attenuation_max

    @property
    def n_attenuation_bins(self) -> int:
        """The number of symmetric attenuation bins."""
        return self._n_attenuation_bins

    @property
    def delay_max(self) -> float:
        """The maximum magnitude of delay to consider."""
        return self._delay_max

    @property
    def n_delay_bins(self) -> int:
        """The number of delay bins."""
        return self._n_delay_bins

    @property
    def max_peaks(self) -> int|None:
        """The maximum number of peaks to find, or None for all peaks."""
        return self._max_peaks

    @property
    def height(self) -> float:
        """Minimum height of peaks."""
        return self._height

    @property
    def footprint(self) -> ndarray:
        """The footprint of the peak-finding algorithm."""
        return self._footprint

    def __init__(self, sample_rate: int = 16000, *, window: int|ndarray = 256,
                 attenuation_max: float = 0.7, n_attenuation_bins: int = 35,
                 delay_max: float = 3.6, n_delay_bins: int = 50,
                 max_peaks: int|None = None, height: float = 50.0, footprint: ndarray|int = 5,
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
        attenuation_max : float
            The maximum magnitude of symmetric attenuation to consider, default is 0.7.
        n_attenuation_bins : int
            The number of symmetric attenuation bins, default is 35.
        delay_max : float
            The maximum magnitude of delay to consider, default is 3.6.
        n_delay_bins : int
            The number of delay bins, default is 50.
        max_peaks : int|None
            The maximum number of peaks to find. If None, all peaks will be returned.
        height : float
            Minimum height of peaks. This is used to filter out small peaks.
            Larger values will result in fewer peaks. Default is 50.
        footprint : ndarray|int
            The footprint of the peak-finding algorithm. This is used to filter
            nearby peaks. Larger values will result in fewer peaks. Default
            is a disk of size 5.
        p : float
            The symmetric attenuation estimator value weights, default is 1.
        q : float
            The delay estimator value weights, default is 0.
        """
        super().__init__(sample_rate=sample_rate, window=window, p=p, q=q)
        self._attenuation_max = attenuation_max
        self._n_attenuation_bins = n_attenuation_bins
        self._delay_max = delay_max
        self._n_delay_bins = n_delay_bins
        self._max_peaks = max_peaks
        self._height = height
        self._footprint = morphology.disk(footprint) if isinstance(footprint, int) else footprint


    def _find_peaks(self, tf_weights: ndarray, sym_atn: ndarray, delay: ndarray
                    ) -> tuple[ndarray, ndarray]:
        hist = self._compute_hist(tf_weights, sym_atn, delay)
        return self._find_hist_peaks(hist)


    def _compute_hist(self, weights: ndarray,
                      alpha: ndarray, delta: ndarray) -> ndarray:
        """
        Calculate histogram. Builds a 2D histogram (phase vs amplitude) where the
        height is the count of time-frequency bins that have approximately that
        phase/amplitude. It is smoothed with a 3x3 mean filter. Last part of step
        3 in the paper.

        Arguments
        ---------
        weights : ndarray
            The weights of each alpha/delta, has shape (f, t)
        alpha : ndarray
            The symmetric attenuation, has shape (f, t)
        delta : ndarray
            The relative delay, has shape (f, t)

        Returns
        -------
        hist : ndarray
            2D histogram of symmetric attenuation and delay, has shape
            (n_attenuation_bins, n_delay_bins)
        """
        a_max, d_max = self.attenuation_max, self.delay_max

        # only consider time-freq points yielding estimates in bounds
        mask = (np.abs(alpha) < a_max) & (np.abs(delta) < d_max)

        # create the 2D histogram (the two other return values are the bin edges)
        hist, _, _ = np.histogram2d(alpha[mask], delta[mask],
                                    [self.n_attenuation_bins, self.n_delay_bins],
                                    [[-a_max, a_max], [-d_max, d_max]],
                                    weights=weights[mask])

        # smooth the histogram - local average 3-by-3 neighboring bins
        hist = sp.ndimage.uniform_filter(hist, 3, mode='reflect')

        return hist


    def _find_hist_peaks(self, hist: ndarray) -> tuple[ndarray, ndarray]:
        """
        Find the peaks in the attenuation-delay histogram. Step 4 in the paper.
        
        Arguments
        ---------
        hist : ndarray
            2D histogram of symmetric attenuation and delay, has shape
            (n_attenuation_bins, n_delay_bins)

        Returns
        -------
        atn_peaks : ndarray
            Peaks of attenuation, has shape (n_peaks,)
        delay_peaks : ndarray
            Peaks of delay, has shape (n_peaks,)
        """
        # NOTE: Original version from Python (but this easily misses peaks since it is actually 1D)
        # delay_side = np.max(hist, axis=0)
        # d_max_idx, prop = sp.signal.find_peaks(delay_side, width=width, prominence=prominence)
        # if self.max_peaks is not None:
        #     prom_rank = np.argsort(prop['prominences'])[::-1][:self.max_peaks]
        #     d_max_idx = d_max_idx[prom_rank]
        # a_max_idx = np.argmax(hist[:, d_max_idx], axis=0)

        a_max_idx, d_max_idx = np.where(morphology.h_maxima(hist, self.height, self.footprint))
        if self.max_peaks is not None:
            a_max_idx = a_max_idx[:self.max_peaks]
            d_max_idx = d_max_idx[:self.max_peaks]

        # get peak values
        a_max, n_a_bins = self.attenuation_max, self.n_attenuation_bins
        d_max, n_d_bins = self.delay_max, self.n_delay_bins
        # TODO: this uses bin left edges, but maybe bin centers would be better?
        #   need to + 0.5 * (2 * a_max / n_a_bins)
        sym_atn_peaks = a_max_idx * (2 * a_max / (n_a_bins - 1)) - a_max
        delay_peaks = d_max_idx * (2 * d_max / (n_d_bins - 1)) - d_max

        return sym_atn_peaks, delay_peaks
