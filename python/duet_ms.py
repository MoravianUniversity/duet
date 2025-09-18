"""
DUET Algorithm with Mean Shifting.

The paper can be found at:
https://www.sciencedirect.com/science/article/abs/pii/S0165168412000722

This algorithm uses the same first and last steps as the original DUET algorithm
but steps 3 and 4 (construction of the weighted histogram and finding the peaks)
are replaced with a specialized weighted mean-shift algorithm.

Improvements that could be made:
 * Same as for DUET:
    * "big delay" support
    * removal of scaling factors
    * general optimizations
 * Better "online" support (recomputes everything from scratch each time)
 * Alternative seed generation methods (like using the previous mean-shift
   centroids along with some grid points from new data)

Should Tune:
 * All of the parameters of the __init__ method, particularly the window length
   and bandwidth (the paper uses 0.5 & 0.8 but those are absurdly large, so they
   must be in some other units) and attenuation/delay max values (symmetric?).
 * Convergence tolerance (currently 0.1*0.2 but that is somewhat arbitrary,
   scikit-learn uses 0.001). Larger goes faster but can result in slightly off
   centroids unlikely to be a problem (the original DUET paper rounded
   everything to 0.04 and 0.144).
"""

from functools import cache
from typing import Sequence

import numpy as np
from numpy import ndarray

from duet_base import DuetBase
from mean_shift import mean_shift, make_gaussian_kernel, get_seeds


class DuetMS(DuetBase):
    """
    DUET with Mean-Shifting algorithm implementation.
    """
    _force_stereo = False  # supports multichannel data

    @property
    def threshold(self) -> float:
        """
        The threshold to filter the points in the spectrogram.
        The higher this value, the faster it will run, but it may also start
        moving the cluster centers around.
        """
        return self._threshold

    @property
    def bandwidth(self) -> float|Sequence[float]:
        """
        The bandwidth of the Gaussian kernel used in the mean-shift algorithm.
        Can be a single value, a sequence of two values for each of alpha and
        delta, or many alphas and deltas for multichannel data. Larger values
        go faster but can easily start to merge clusters. Too small and it will
        begin to find lots of local minima.
        """
        return self._bandwidth

    @property
    def attenuation_max(self) -> float:
        """The maximum magnitude of symmetric attenuation to consider."""
        return self._attenuation_max

    @property
    def delay_max(self) -> float:
        """The maximum magnitude of delay to consider."""
        return self._delay_max

    @property
    def seed_count(self) -> int|None:
        """
        The number of seeds to consider for mean-shift.
        If None, then all points are considered.
        Smaller values go faster but can result in missing clusters.
        """
        return self._seed_count

    @property
    def min_bin_count(self) -> int:
        """
        The minimum number of points in a bin to consider it as a seed.
        Larger values go faster but can result in missing clusters.
        """
        return self._min_bin_count

    @property
    def convergence_tol(self) -> float:
        """
        The convergence tolerance for the mean-shift algorithm.
        Larger values go faster but can result in slightly off centroids.
        """
        return self._convergence_tol

    def __init__(self, sample_rate: int = 16000, *, window: int|ndarray = 256,
                 threshold: float = 0.05, bandwidth: float|Sequence[float] = 0.2,
                 attenuation_max: float = 0.7, delay_max: float = 3.6,
                 seed_count: int|None = 25, min_bin_count: int = 1,
                 convergence_tol: float = 0.1,
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
        threshold : float
            The threshold to filter the points in the spectrogram. The higher this value,
            the faster it will run, but it may also start moving the cluster centers around.
            Default is 0.05.
        bandwidth : float|Sequence[float]
            The bandwidth of the Gaussian kernel used in the mean-shift algorithm. Can be a single
            value, a sequence of two values for each of alpha and delta, or many alphas and deltas
            for multichannel data. Larger values go faster but can easily start to merge clusters.
            Too small and it will begin to find lots of local minima. Default is 0.2.
        attenuation_max : float
            The maximum magnitude of symmetric attenuation to consider during seed generation,
            default is 0.7.
        delay_max : float
            The maximum magnitude of delay to consider during seed generation, default is 3.6.
        seed_count : int|None
            Number of seeds to consider for mean-shift. If None, then all points are considered.
            Smaller values go faster but can result in missing clusters. Default is 25.
        min_bin_count : int
            The minimum number of points in a bin to consider it as a seed.
            Larger values go faster but can result in missing clusters. Default is 1.
        convergence_tol : float
            The convergence tolerance for the mean-shift algorithm.
            Larger values go faster but can result in slightly off centroids.
            The default (0.1) is somewhat arbitrary (scikit-learn uses 0.001).
        p : float
            The symmetric attenuation estimator value weights, default is 1.
        q : float
            The delay estimator value weights, default is 0.
        """
        super().__init__(sample_rate=sample_rate, window=window, p=p, q=q)
        self._threshold = threshold
        self._bandwidth = bandwidth
        self._attenuation_max = attenuation_max
        self._delay_max = delay_max
        self._seed_count = seed_count
        self._min_bin_count = min_bin_count
        self._convergence_tol = convergence_tol


    def _find_peaks(self, tf_weights: ndarray, sym_atn: ndarray, delay: ndarray
                    ) -> tuple[ndarray, ndarray]:
        n = tf_weights.shape[0] if tf_weights.ndim == 3 else 1
        points, weights = self._get_points(tf_weights, sym_atn, delay)
        bandwidths = self._bandwidths(n)
        seeds = get_seeds(points, bandwidths, self.min_bin_count, self.seed_count, self._bounds(n))
        if seeds.size == 0:
            # No seeds found, return empty arrays
            empty = np.empty((weights.shape[0], 0)) if tf_weights.ndim == 3 else np.empty((0,))
            return empty, empty
        kernel = make_gaussian_kernel(bandwidths, weights)
        centroids = mean_shift(points, kernel, seeds, np.min(bandwidths)).T
        if tf_weights.ndim == 3:
            half = len(centroids)//2
            return centroids[:half, :], centroids[half:, :]
        else:
            return centroids[0, :], centroids[1, :]


    def _get_points(self, weights: ndarray, alpha: ndarray, delta: ndarray
                    ) -> tuple[ndarray, ndarray]:
        """
        Get the points weights for the mean-shift algorithm. This filters the
        points based on the weights and the threshold.

        Arguments
        ---------
        weights : ndarray
            The weights of the points, has shape (f, t) or (n_channels-1, f, t)
        alpha : ndarray
            The symmetric attenuation of the points, has shape (f, t) or
            (n_channels-1, f, t)
        delta : ndarray
            The relative delay of the points, has shape (f, t) or
            (n_channels-1, f, t)

        Returns
        -------
        points : ndarray
            The points to use for the mean-shift algorithm, has shape
            (n, 2) or (n, 2*n_channels-2). The first half of the columns are
            the symmetric attenuation and the second half are the relative
            delay.
        weights : ndarray
            The weights of the points, has shape (n,)
        """
        # TODO: lots of transpose and reshape here, maybe we can do better
        # (maybe do the transpose of everything in the mean-shift functions)

        # Reshape the data to be n-by-tf
        n = weights.shape[0] if weights.ndim == 3 else 1
        tf = weights.shape[-2] * weights.shape[-1]
        alpha = alpha.reshape(n, tf)
        delta = delta.reshape(n, tf)
        points = np.concatenate((alpha, delta))

        # Get the weights
        if weights.ndim == 3:
            weights = weights.reshape(-1, tf).product(axis=0) # TODO: check this (maybe max?)
        else:
            weights = weights.reshape(tf)

        # Reduce the number of points to consider to speed up the process
        mask = weights > self.threshold
        # NOTE: should test this again, but it seems like the threshold method is more robust; even
        # without masking, these are already eliminated from the seeds, just not mean shift itself
        mask &= (np.abs(alpha) < self.attenuation_max).all(0) & (np.abs(delta) < self.delay_max).all(0)
        points = points[:, mask]
        weights = weights[mask]

        return points.T, weights


    @cache
    def _bounds(self, n: int = 1) -> ndarray:
        a_max, d_max = self.attenuation_max, self.delay_max
        bounds = [[-a_max, a_max]]*n + [[-d_max, d_max]]*n
        return np.array(bounds)


    @cache
    def _bandwidths(self, n: int = 1) -> float|ndarray:
        if isinstance(self.bandwidth, float):
            return self.bandwidth
        if len(self.bandwidth) == 2:
            return np.array(self.bandwidth) if n == 1 else np.asarray(self.bandwidth).repeat(n)
        if len(self.bandwidth) != 2*n:
            raise ValueError(f"Invalid bandwidth shape: {self.bandwidth}")
        return np.asarray(self.bandwidth)
