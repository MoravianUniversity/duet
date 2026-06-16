"""
DUET Algorithm with Mean Shifting.

The paper can be found at:
https://www.sciencedirect.com/science/article/abs/pii/S0165168412000722

This algorithm uses the same first and last steps as the original DUET algorithm
but steps 3 and 4 (construction of the weighted histogram and finding the peaks)
are replaced with a specialized weighted mean-shift algorithm.

Improvements that could be made:
 * Same as for DUET
 * Better "online" support (recomputes everything from scratch each time)
 * Alternative seed generation methods (like using the previous mean-shift
   centroids along with some grid points from new data, or a simple peak-finding
   method to find local maxima in the data)

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
from collections.abc import Sequence

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
    def alpha_max(self) -> float:
        """The maximum magnitude of (symmetric) attenuation to consider."""
        return self._alpha_max

    @property
    def delta_max(self) -> float:
        """The maximum magnitude of relative delay to consider."""
        return self._delta_max

    @property
    def time_bandwidth(self) -> float:
        """
        The bandwidth to use for the time dimension. If this is <=0, then time
        is not included in the mean-shift algorithm. The time values are in
        units of the STFT hop size, so a value of 1 would mean that points
        within 1 hop of each other in time would be considered close.
        """
        return self._time_bandwidth
    
    @property
    def freq_bandwidth(self) -> float:
        """
        The bandwidth to use for the frequency dimension. If this is <=0, then
        frequency is not included in the mean-shift algorithm. The frequency
        values are in units of the STFT bin size, so a value of 1 would mean
        that points within 1 bin of each other in frequency would be considered
        close.
        """
        return self._freq_bandwidth

    @property
    def seed_count(self) -> int|None:
        """
        The number of seeds to consider for mean-shift.
        If None, then all points are considered.
        Smaller values go faster but can result in missing clusters.
        """
        return self._seed_count

    @property
    def min_bin_count(self) -> int|float:
        """
        The minimum number of points in a bin to consider it as a seed.
        Larger values go faster but can result in missing clusters. When computing seeds using
        weights, this is based on the weighted count of the bins instead of the number of points
        in the bins, so it can be a float value less than 1 to allow for bins that have some
        weight but not a full point.
        """
        return self._min_bin_count
    
    @property
    def max_filter_size(self) -> tuple[int]|int|None:
        """
        The maximum filter size to use for mean-shift seed selection.
        Must be None (for no filtering) or odd integers >1 for filtering.
        Remove possible seeds that are not local maxima within max_filter_size; this can help speed
        up results a lot by removing seeds. As this is increased, seed_count should be decreased or
        min_bin_count increased to prevent finding random local maxima that are not sources.
        """
        return self._max_filter_size

    @property
    def compute_seeds_using_weights(self) -> bool:
        """
        Whether to compute the seeds using the weights or not. This can be useful to speed up
        results by only considering bins that have a high weight. This effects the `min_bin_count`
        parameter which is based on the weighted count of the bins instead of the number of points
        in the bins.
        """
        return self._compute_seeds_using_weights

    @property
    def convergence_tol(self) -> float:
        """
        The convergence tolerance for the mean-shift algorithm.
        Larger values go faster but can result in slightly off centroids.
        """
        return self._convergence_tol

    def __init__(self, sample_rate: int = 16000, *, window: int|ndarray = 256, oversample: int = 1,
                 threshold: float = 0.05, bandwidth: float|Sequence[float] = 0.2,
                 alpha_max: float = 0.7, delta_max: float = 3.6,
                 time_bandwidth: float = 0.0, freq_bandwidth: float = 0.0,
                 seed_count: int|None = 25, min_bin_count: int = 1,
                 max_filter_size: tuple[int]|int|None = None,
                 compute_seeds_using_weights: bool = False,
                 convergence_tol: float = 0.1,
                 alpha_op: str = "symmetric", big_delay: str = "none",
                 delta_smoothing: tuple[int, int] = (1, 1), delta_smoothing_mode: str = "mean",
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
        oversample : int
            The oversampling factor for the STFT. Larger values will result in better time
            resolution but worse frequency resolution. Default is 1 (no oversampling).
        threshold : float
            The threshold to filter the points in the spectrogram. The higher this value,
            the faster it will run, but it may also start moving the cluster centers around.
            Default is 0.05.
        bandwidth : float|Sequence[float]
            The bandwidth of the Gaussian kernel used in the mean-shift algorithm. Can be a single
            value, a sequence of two values for each of alpha and delta, or many alphas and deltas
            for multichannel data. Larger values go faster but can easily start to merge clusters.
            Too small and it will begin to find lots of local minima. Default is 0.2.
        alpha_max : float
            The maximum magnitude of (symmetric) attenuation to consider during seed generation,
            default is 0.7.
        delta_max : float
            The maximum magnitude of delay to consider during seed generation, default is 3.6.
        time_bandwidth : float
            The bandwidth to use for the time dimension. If this is <=0, then time is not included
            in the mean-shift algorithm. The time values are in units of the STFT hop size, so a
            value of 1 would mean that points within 1 hop of each other in time would be
            considered close. Can be useful for separating sources that have similar
            attenuation/delay but are active at different times, default is 0 (not included).
        freq_bandwidth : float
            The bandwidth to use for the frequency dimension. If this is <=0, then frequency is not
            included in the mean-shift algorithm. The frequency values are in units of the STFT bin
            size, so a value of 1 would mean that points within 1 bin of each other in frequency
            would be considered close. Can be useful for separating sources that have similar
            attenuation/delay but are active at different frequencies, default is 0 (not included).
        seed_count : int|None
            Number of seeds to consider for mean-shift. If None, then all points are considered.
            Smaller values go faster but can result in missing clusters. Default is 25.
        min_bin_count : int
            The minimum number of points in a bin to consider it as a seed.
            Larger values go faster but can result in missing clusters. Default is 1.
        max_filter_size : tuple[int]|int|None
            The maximum filter size to use for mean-shift seed selection. Must be None (for no
            filtering) or odd integers >1 for filtering. Default is None.
        compute_seeds_using_weights : bool
            Whether to compute the seeds using the weights or not. This can be useful to speed up
            results by only considering bins that have a high weight. This effects the
            `min_bin_count` parameter which is based on the weighted count of the bins instead of
            the number of points in the bins. Default is False.
        convergence_tol : float
            The convergence tolerance for the mean-shift algorithm.
            Larger values go faster but can result in slightly off centroids.
            The default (0.1) is somewhat arbitrary (scikit-learn uses 0.001).
        alpha_op : str
            The type of alpha operation to use, can be "symmetric" (a-1/a), "log" (log a), or
            "none". Default is "symmetric".
        big_delay : str
            The type of big delay algorithm to use, can be "diff" or "none". Default is "none".
        delta_smoothing : tuple[int, int]
            The size of the smoothing filter for the delay estimator, as a tuple of (freq, time).
            Default is (1, 1) (no smoothing).
        delta_smoothing_mode : str
            The type of smoothing to apply to the delay estimator, can be "mean", "median", or
            "gaussian". Default is "mean".
        p : float
            The symmetric attenuation estimator value weights, default is 1.
        q : float
            The delay estimator value weights, default is 0.
        """
        super().__init__(sample_rate=sample_rate, window=window, oversample=oversample,
                         alpha_op=alpha_op, big_delay=big_delay, delta_smoothing=delta_smoothing,
                         delta_smoothing_mode=delta_smoothing_mode, p=p, q=q)
        self._threshold = threshold
        self._bandwidth = bandwidth
        self._alpha_max = alpha_max
        self._delta_max = delta_max
        self._time_bandwidth = time_bandwidth
        self._freq_bandwidth = freq_bandwidth
        self._seed_count = seed_count
        self._min_bin_count = min_bin_count
        self._max_filter_size = max_filter_size
        self._compute_seeds_using_weights = compute_seeds_using_weights
        self._convergence_tol = convergence_tol


    def _find_peaks(self, tf_weights: ndarray, alpha: ndarray, delta: ndarray,
                    ) -> tuple[ndarray, ndarray]:
        n = tf_weights.shape[0] if tf_weights.ndim == 3 else 1
        points, weights = self._get_points(tf_weights, alpha, delta)
        bandwidths = self._bandwidths(n)
        seeds = get_seeds(points, bandwidths,
                          weights=weights if self.compute_seeds_using_weights else None,
                          min_count=self.min_bin_count,
                          top_n=self.seed_count,
                          max_filter_size=self.max_filter_size,
                          bounds=self._bounds(n, tf_weights.shape[-2:]))
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


    def _get_points(self, weights: ndarray, alpha: ndarray, delta: ndarray,
                    ) -> tuple[ndarray, ndarray]:
        """
        Get the points weights for the mean-shift algorithm. This filters the
        points based on the weights and the threshold and includes the time and
        frequency information if specified.

        Arguments
        ---------
        weights : ndarray
            The weights of the points, has shape (f, t) or (n_channels-1, f, t)
        alpha : ndarray
            The (symmetric) attenuation of the points, has shape (f, t) or
            (n_channels-1, f, t)
        delta : ndarray
            The relative delay of the points, has shape (f, t) or
            (n_channels-1, f, t)

        Returns
        -------
        points : ndarray
            The points to use for the mean-shift algorithm, has shape
            (n, 2*n_channels-2+include_time+include_freq). The first half of
            the columns are the (symmetric) attenuation and the second half are
            the relative delay. The last columns are the time and frequency
            values, if specified.
        weights : ndarray
            The weights of the points, has shape (n,)
        """
        # TODO: lots of transpose and reshape here, maybe we can do better
        # (maybe do the transpose of everything in the mean-shift functions)

        # Reshape the data to be n-by-tf
        n = weights.shape[0] if weights.ndim == 3 else 1
        f, t = weights.shape[-2:]
        tf = f * t
        alpha = alpha.reshape(n, tf)
        delta = delta.reshape(n, tf)
        pts = (alpha, delta)
        if self.time_bandwidth > 0:
            pts += (np.tile(np.arange(t), f).reshape(1, -1),)
        if self.freq_bandwidth > 0:
            pts += (np.repeat(np.arange(f), t).reshape(1, -1),)
        points = np.concatenate(pts)

        # Get the weights
        if weights.ndim == 3:
            weights = weights.reshape(-1, tf).product(axis=0) # TODO: check this (maybe max?)
        else:
            weights = weights.reshape(tf)

        # Reduce the number of points to consider to speed up the process
        mask = weights > self.threshold
        # NOTE: the threshold method is more robust than clipping the values, however to speed up
        # the C code, it requires clipping the values to be within the alpha and delta max values,
        # so we do that here as well
        mask &= (np.abs(alpha) < self.alpha_max).all(0) & (np.abs(delta) < self.delta_max).all(0)
        points = points[:, mask]
        weights = weights[mask]

        return points.T, weights


    @cache
    def _bounds(self, n: int = 1, tf_shape: tuple[int, int] = (0, 0)) -> ndarray:
        a_max, d_max = self.alpha_max, self.delta_max
        bounds = [[-a_max, a_max]]*n + [[-d_max, d_max]]*n
        if self.time_bandwidth > 0:
            bounds += [[0, tf_shape[1]]]
        if self.freq_bandwidth > 0:
            bounds += [[0, tf_shape[0]]]
        return np.array(bounds)


    @cache
    def _bandwidths(self, n: int = 1) -> float|ndarray:
        tf_bandwidth = [self.time_bandwidth] if self.time_bandwidth > 0 else []
        tf_bandwidth += [self.freq_bandwidth] if self.freq_bandwidth > 0 else []
        bandwidths = None
        if isinstance(self.bandwidth, float):
            return np.array([self.bandwidth]*(2*n) + tf_bandwidth) if tf_bandwidth else self.bandwidth
        if len(self.bandwidth) == 2:
            bandwidths = np.array(self.bandwidth) if n == 1 else np.asarray(self.bandwidth).repeat(n)
        if len(self.bandwidth) == 2*n:
            bandwidths = np.asarray(self.bandwidth)
        if bandwidths is None:
            raise ValueError(f"Invalid bandwidth shape: {self.bandwidth}")
        return np.concatenate([bandwidths, tf_bandwidth]) if tf_bandwidth else bandwidths
