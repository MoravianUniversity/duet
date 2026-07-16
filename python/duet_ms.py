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
from mean_shift import mean_shift, make_gaussian_kernel, estimate_bandwidth, bandwidths_to_iso
from mean_shift import build_histogram, get_seeds_from_histogram, points2coords
from mean_shift import compute_pilot_density, make_dynamic_gaussian_kernel, make_adaptive_gaussian_kernel


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
    def bandwidth_mode(self) -> str:
        """
        The bandwidth mode to use for the mean-shift algorithm. Can be:
        
        "fixed" -> always use the given bandwidth for all points
        "adaptive" -> adapt the bandwidth based on the local density of points, but still uses the
                      bandwidth as a global scale factor
        "dynamic" -> adaptive plus update the bandwidth continuously during convergence
        """
        return self._bandwidth_mode

    @property
    def bandwidth(self) -> float|str|Sequence[float|str]:
        """
        The bandwidth of the Gaussian kernel used in the mean-shift algorithm.
        Can be a single value, a sequence of two values for each of alpha and
        delta, or many alphas and deltas for multichannel data. Larger values
        go faster but can easily start to merge clusters. Too small and it will
        begin to find lots of local minima.

        Can also be a string of "silverman" to use the Silverman rule of thumb
        to estimate the bandwidth from the data for each dimension. Using
        "iso-silverman" will use the same bandwidth for all dimensions.
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
    def bin_size(self) -> float|str|Sequence[float|str]|None:
        """
        The size of each bin selecting seeds for the mean-shift algorithm. By default this will use
        a fraction of the bandwidth (see `bin_size_frac`), but it can be set to exact values or
        "silverman" or "iso-silverman".
        """
        return self._bin_size

    @property
    def bin_size_frac(self) -> float|Sequence[float]:
        """
        If `bin_size` is `None`, then this is the fraction of the bandwidth to use
        for the bin size. Default is 1.0 (same as `bandwidth`).
        """
        return self._bin_size_frac

    @property
    def density_floor_factor(self) -> float:
        """
        The density floor factor to use for the pilot density estimation. This is used to prevent
        the pilot density from being too small, which can cause the adaptive/dynamic bandwidth to
        be too small and result in overfitting. Default is 0.75. Not used when bandwidth_mode is
        "fixed".
        """
        return self._density_floor_factor

    @property
    def lambda_min(self) -> float:
        """
        The minimum lambda value to use for the adaptive/dynamic bandwidth. This is used to prevent
        the adaptive/dynamic bandwidth from being too small, which can cause overfitting. Default
        is 0.2. Not used when bandwidth_mode is "fixed".
        """
        return self._lambda_min

    @property
    def lambda_max(self) -> float:
        """
        The maximum lambda value to use for the adaptive/dynamic bandwidth. This is used to prevent
        the adaptive/dynamic bandwidth from being too large, which can cause underfitting. Default
        is 5.0. Not used when bandwidth_mode is "fixed".
        """
        return self._lambda_max

    @property
    def silverman_factor(self) -> float:
        """
        The factor to use for the Silverman rule of thumb to estimate the bandwidth from the data
        for each dimension (or all dimensions). This is only used when `bandwidth` or `bin_size` is
        set to "silverman" or "iso-silverman". Default is 2.0. The value 4.0 is the original value.
        """
        return self._silverman_factor

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
                 threshold: float = 0.05,
                 bandwidth_mode: str = "fixed",
                 bandwidth: float|str|Sequence[str|float] = 0.2,
                 alpha_max: float = 0.7, delta_max: float = 3.6,
                 seed_count: int|None = 25, min_bin_count: int = 1,
                 max_filter_size: tuple[int]|int|None = None,
                 bin_size: float|str|Sequence[float|str]|None = None,
                 bin_size_frac: float|Sequence[float] = 1.0,
                 density_floor_factor: float = 0.75,
                 lambda_min: float = 0.2, lambda_max: float = 5.0,
                 silverman_factor: float = 2.0,
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
        bandwidth_mode : str
            The mode to use for the bandwidth. Can be "fixed", "adaptive", or "dynamic". Default is
            "fixed" which uses the given bandwidth. For "adaptive", the bandwidth is adapted based
            on the local density of points, but still uses the bandwidth as a global scale factor.
            For "dynamic", the bandwidth is adaptive and is updated continuously during convergence.
        bandwidth : float|str|Sequence[float|str]
            The bandwidth of the Gaussian kernel used in the mean-shift algorithm. Can be a single
            value, a sequence of two values for each of alpha and delta, or many alphas and deltas
            for multichannel data. Larger values go faster but can easily start to merge clusters.
            Too small and it will begin to find lots of local minima. Default is 0.2. This also
            supports "silverman" and "iso-silverman" to use the Silverman rule of thumb to estimate
            the bandwidth from the data for each dimension (or all dimensions).
        alpha_max : float
            The maximum magnitude of (symmetric) attenuation to consider during seed generation,
            default is 0.7.
        delta_max : float
            The maximum magnitude of delay to consider during seed generation, default is 3.6.
        seed_count : int|None
            Number of seeds to consider for mean-shift. If None, then all points are considered.
            Smaller values go faster but can result in missing clusters. Default is 25.
        min_bin_count : int
            The minimum number of points in a bin to consider it as a seed.
            Larger values go faster but can result in missing clusters. Default is 1.
        max_filter_size : tuple[int]|int|None
            The maximum filter size to use for mean-shift seed selection. Must be None (for no
            filtering) or odd integers >1 for filtering. Default is None.
        bin_size : float|str|Sequence[float|str]|None
            The size of each bin selecting seeds for the mean-shift algorithm. By default this uses
            a fraction of the bandwidth (see `bin_size_frac`), but it can be set to specific values
            or "silverman" or "iso-silverman". Default is `None` (use fraction of bandwidth).
        bin_size_frac : float|Sequence[float]
            If `bin_size` is `None`, then this is the fraction of the bandwidth to use for the bin
            size. Default is 1.0 (same as `bandwidth`).
        density_floor_factor : float
            The density floor factor to use for the pilot density estimation. This is used to
            prevent the pilot density from being too small, which can cause the adaptive/dynamic
            bandwidth to be too small and result in overfitting. Default is 0.75. Not used when
            bandwidth_mode is "fixed".
        lambda_min : float
            The minimum lambda value to use for the adaptive/dynamic bandwidth. This is used to
            prevent the adaptive/dynamic bandwidth from being too small, which can cause
            overfitting. Default is 0.2. Not used when bandwidth_mode is "fixed".
        lambda_max : float
            The maximum lambda value to use for the adaptive/dynamic bandwidth. This is used to
            prevent the adaptive/dynamic bandwidth from being too large, which can cause
            underfitting. Default is 5.0. Not used when bandwidth_mode is "fixed".
        silverman_factor : float
            The factor to use for the Silverman rule of thumb to estimate the bandwidth from the
            data for each dimension (or all dimensions). This is only used when `bandwidth` or
            `bin_size` is set to "silverman" or "iso-silverman".
            Default is 2.0. The value 4.0 is the original value.
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
        if bandwidth_mode not in ("fixed", "adaptive", "dynamic"):
            raise ValueError(f"Invalid bandwidth_mode: {bandwidth_mode}. Must be 'fixed', 'adaptive', or 'dynamic'.")
        self._bandwidth_mode = bandwidth_mode
        if not self.__is_valid_bandwidth(bandwidth):
            raise ValueError(f"Invalid bandwidth: {bandwidth}. Must be a positive number or 'silverman' or 'iso-silverman', or an even sequence of such values.")
        self._bandwidth = bandwidth
        self._alpha_max = alpha_max
        self._delta_max = delta_max
        self._seed_count = seed_count
        self._min_bin_count = min_bin_count
        self._max_filter_size = max_filter_size
        self._bin_size = bin_size
        self._bin_size_frac = bin_size_frac
        self._density_floor_factor = density_floor_factor
        self._lambda_min = lambda_min
        self._lambda_max = lambda_max
        self._silverman_factor = silverman_factor
        self._compute_seeds_using_weights = compute_seeds_using_weights
        self._convergence_tol = convergence_tol


    def _find_peaks(self, tf_weights: ndarray, alpha: ndarray, delta: ndarray,
                    ) -> tuple[ndarray, ndarray]:
        n = tf_weights.shape[0] if tf_weights.ndim == 3 else 1
        points, weights = self._get_points(tf_weights, alpha, delta)
        bandwidths = self._compute_bandwidths(self.bandwidth, points, weights, n)

        # Compute the initial seeds
        if self.bin_size is None:
            bin_sizes = np.asarray(self._expand_bandwidths(self.bin_size_frac, n)) * np.asarray(bandwidths)
        else:
            bin_sizes = np.asarray(self._compute_bandwidths(self.bin_size, points, weights, n))

        hist, _, mins = build_histogram(points, bin_sizes, weights if self.compute_seeds_using_weights else None,
                                        self._bounds(n, tf_weights.shape[-2:]))
        if self.compute_seeds_using_weights:
            hist_weighted = hist
        elif self.bandwidth_mode != "fixed":
            # If not using weights, then we need to compute the weighted histogram for adaptive/dynamic bandwidths
            hist_weighted, _, _ = build_histogram(points, bin_sizes, weights, self._bounds(n, tf_weights.shape[-2:]))
        else:
            hist_weighted = None

        # Get the seeds from the histogram
        raw_seeds = get_seeds_from_histogram(hist, min_count=self.min_bin_count,
                                             top_n=self.seed_count, max_filter_size=self.max_filter_size)
        seeds = points2coords(raw_seeds, bin_sizes, mins)
        if seeds.size == 0:
            # No seeds found, return empty arrays
            empty = np.empty((weights.shape[0], 0)) if tf_weights.ndim == 3 else np.empty((0,))
            return empty, empty

        # Determine the kernel
        if self.bandwidth_mode == "fixed":
            kernel = make_gaussian_kernel(bandwidths, weights)
        elif self.bandwidth_mode == "adaptive":
            pilot, g = compute_pilot_density(hist_weighted, bin_sizes, density_floor_factor=self._density_floor_factor)
            kernel = make_adaptive_gaussian_kernel(bandwidths, pilot, g, bin_sizes, mins, weights,
                                                   lambda_min=self._lambda_min, lambda_max=self._lambda_max)
            pass
        elif self.bandwidth_mode == "dynamic":
            pilot, g = compute_pilot_density(hist_weighted, bin_sizes, density_floor_factor=self._density_floor_factor)
            kernel = make_dynamic_gaussian_kernel(bandwidths, pilot, g, bin_sizes, mins, weights,
                                                  lambda_min=self._lambda_min, lambda_max=self._lambda_max)
        else:
            raise ValueError(f"Invalid bandwidth_mode: {self.bandwidth_mode}. Must be 'fixed', 'adaptive', or 'dynamic'.")

        # Run the mean-shift algorithm
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
        return np.array(bounds)

    @staticmethod
    def _compute_bandwidths(value: float|int|str|Sequence[float|str], points: ndarray, weights: ndarray, n: int = 1) -> float|ndarray:
        """
        Compute the bandwidths for the mean-shift algorithm. This is a helper
        function to compute the bandwidths based on the bandwidth raw values
        (which can include special strings) and the number of channels.

        Arguments
        ---------
        value : float|int|str|Sequence[float|str]
            The bandwidth raw values, can be a single value, a sequence of two
            values for each of alpha and delta, or many alphas and deltas for
            multichannel data. Can also be "silverman" or "iso-silverman" to
            use the Silverman rule of thumb to estimate the bandwidth from the
            data for each dimension (or all dimensions).
        points : ndarray
            The points to use for the mean-shift algorithm, has shape
            (n, 2*n_channels-2+include_time+include_freq).
        weights : ndarray
            The weights of the points, has shape (n,)
        n : int
            The number of channel pairs.

        Returns
        -------
        bandwidths : float|ndarray
            The bandwidths to use for the mean-shift algorithm, a single scalar, or array with shape
            (2*n+include_time+include_freq,)
        """
        if isinstance(value, (float, int)) or isinstance(value, Sequence) and all(isinstance(b, (float, int)) for b in value):
            return np.asarray(DuetMS._expand_bandwidths(value, n))
        elif isinstance(value, str):
            bandwidths, silverman_factor = estimate_bandwidth(points, weights)
            if value == "iso-silverman":
                bandwidths = bandwidths_to_iso(silverman_factor, bandwidths)
            return bandwidths
        else:
            est, silverman_factor = estimate_bandwidth(points, weights)
            iso = bandwidths_to_iso(silverman_factor, est)
            bandwidths = DuetMS._expand_bandwidths(value, n)
            bandwidths = [e if b == "silverman" else iso if b == "iso-silverman" else b
                          for b, e in zip(bandwidths, est)]
            return np.asarray(bandwidths)
    
    @staticmethod
    def _expand_bandwidths(value: float|int|Sequence[float|str], n: int = 1) -> float|Sequence[float|str]:
        if isinstance(value, (float, int)):
            return float(value)
        if len(value) == 2*n:
            return value
        if len(value) == 2:
            return [item for item in value for _ in range(n)]
        raise ValueError(f"Invalid bandwidth shape: {value}")

    @staticmethod
    def __is_valid_bandwidth(bandwidth: float|str|Sequence[float|str]) -> bool:
        if isinstance(bandwidth, (float, int)):
            return bandwidth > 0
        elif isinstance(bandwidth, str):
            return bandwidth in ("silverman", "iso-silverman")
        elif isinstance(bandwidth, Sequence):
            if len(bandwidth) == 0 or len(bandwidth) % 2 != 0:
                return False
            for b in bandwidth:
                if not DuetMS.__is_valid_bandwidth(b):
                    return False
            return True
        else:
            return False
