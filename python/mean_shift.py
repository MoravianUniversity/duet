"""
Adaptive, Dynamic, Weighted Mean Shift

Weighted Mean Shift from https://www.sciencedirect.com/science/article/abs/pii/S0165168412000722
"""

from typing import Sequence, Callable
from collections.abc import Sized

import numpy as np
from numpy import ndarray
from scipy.ndimage import maximum_filter

def mean_shift(points: ndarray,
               kernel: Callable|None = None,
               seeds: ndarray|None = None,
               bandwidth: float = 1.0, # isn't the actual bandwidth, but must be passed to match
               convergence_tol: float = 0.1,
               ) -> ndarray:
    """
    Perform mean shift clustering on the given points.

    Parameters:
    points : ndarray, shape (n_samples, n_features)
        The input data.
    kernel : callable, optional
        The kernel function to use, default is a Gaussian kernel with the given bandwidth
    seeds : array-like, shape (n_samples, n_features), optional
        The initial seeds for the clustering algorithm. If None, the seeds will be generated
        automatically using a grid-based approach.
    bandwidth : float, optional
        The bandwidth of the kernel. This must be the same as the bandwidth used to generate
        the kernel function. This is used to determine the convergence tolerance, grid size,
        and to remove near centroids.
    convergence_tol : float, optional
        The convergence tolerance for the mean shift algorithm. This is the maximum distance
        between the current and new centroid for the algorithm to consider it converged (scaled
        by the bandwidth). Default is 0.1. Larger values go faster but can result in slightly
        off centroids.

    Returns:
    centroids : ndarray, shape (n_centroids, n_features)
        The final centroids after mean shift clustering. The number of centroids may be less than
        the number of initial seeds if some centroids merge to the same location.
    """
    if kernel is None:
        kernel = make_gaussian_kernel(bandwidth)
    if seeds is None:
        seeds = get_seeds(points, bandwidth)
    return _internal_mean_shift(points, kernel, seeds, convergence_tol * bandwidth, bandwidth * 0.75)

def _internal_mean_shift(points: ndarray,
                         kernel: Callable,
                         seeds: ndarray,
                         convergence_tol: float,
                         close_tol: float,
                         ) -> ndarray:
    """
    Perform mean shift clustering on the given points.

    Parameters:
    points : ndarray, shape (n_samples, n_features)
        The input data.
    kernel : callable, optional
        The kernel function to use
    seeds : array-like, shape (n_samples, n_features)
        The initial seeds for the clustering algorithm.
    convergence_tol : float
        The convergence tolerance for the mean shift algorithm. This is the maximum distance
        between the current and new centroid for the algorithm to consider it converged. Larger
        values go faster but can result in slightly off centroids.
    close_tol : float
        The tolerance for considering two centroids to be close enough to merge. This is the
        maximum distance between two centroids for them to be considered the same cluster. Larger
        values will result in fewer clusters, but may merge distinct clusters together. This
        should be on the order of the bandwidth of the kernel, but may need to be adjusted based
        on the specific data and kernel used.

    Returns:
    centroids : ndarray, shape (n_centroids, n_features)
        The final centroids after mean shift clustering. The number of centroids may be less than
        the number of initial seeds if some centroids merge to the same location.
    """
    centroids = np.array(seeds)

    # Squared tolerance so we don't have to take the square root
    tol_2 = convergence_tol * convergence_tol

    # Bad centroids to remove (rare, but does happen)
    remove = []

    for i, centroid in enumerate(centroids):  # can be parallelized
        dist_2 = np.inf
        while dist_2 > tol_2:
            # Shift the centroid
            weights = kernel(centroid, points)
            weights_sum = weights.sum()
            if weights_sum == 0:
                # If no points are near the centroid at all, remove it
                remove.append(i)
                break
            p_new = (weights[:, None] * points).sum(0) / weights_sum

            # Check if the point has converged
            dist_2 = euclidean_distance_2(p_new, centroid)

            # Update the centroid
            centroids[i] = p_new

    # Remove bad centroids
    if len(remove) > 0:
        centroids = np.delete(centroids, remove, axis=0)
        if len(centroids) == 0:
            return centroids

    # Combine all centroids that are close to each other
    return remove_near_duplicates(centroids, close_tol)


##### UTILITY FUNCTIONS #####

def euclidean_distance_2(a: ndarray, b: ndarray) -> ndarray:
    """
    Compute the squared Euclidean distance between two points or sets of points.

    Parameters:
    a : ndarray, shape is either (d,) or (n, d)
        First point or set of points
    b : ndarray, shape is either (d,) or (n, d)
        Second point or set of points

    Returns:
    dist : ndarray
        Squared Euclidean distance. If a and b are both (d,), the result is a
        scalar. Otherwise, the result is (n,).
    """
    diff = a - b
    return np.sum(diff * diff, axis=-1)


def remove_near_duplicates(points: ndarray, tol: float = 1e-7) -> ndarray:
    """
    Remove points that are within a certain tolerance of each other.

    Parameters:
    points : ndarray, shape (n, d)
        Points to be processed
    tol : float
        Tolerance for considering points to be duplicates, default is 1e-7

    Returns:
    new_points : ndarray, shape (m, d)
        Points after removing near duplicates
    """
    # Note: I tried a version that did a first pass with approximate distance but it was slower
    # and much more complicated (it used |delta| < tol/sqrt(2) to determine if points were close
    # enough during the first pass, but that only saves a multiplication per dimension per point)
    new_points = []
    tol_2 = tol * tol
    while points.shape[0] > 0:
        near_points = euclidean_distance_2(points, points[0, :]) <= tol_2
        new_points.append(points[near_points].mean(0))
        points = points[~near_points]
    return np.array(new_points)


def label_points(centroids: ndarray, points: ndarray) -> ndarray:
    """
    Assign each point to the nearest centroid.

    Parameters:
    centroids : ndarray, shape (k, d)
    points : ndarray, shape (n, d)

    Returns:
    groups : ndarray, shape (n,)
        Indices of the nearest centroid for the corresponding point
    """
    try:
        from sklearn.neighbors import BallTree  # pylint: disable=import-outside-toplevel
        return BallTree(centroids, 1, metric='euclidean').query(
            points, return_distance=False, dualtree=True, sort_results=False).squeeze()
    except ImportError:
        group_assignment = np.zeros(len(points), dtype=int)
        for i, point in enumerate(points):
            group_assignment[i] = np.argmin(euclidean_distance_2(point, centroids))
        return group_assignment


def get_seeds(points: ndarray, bin_size, *, weights: ndarray|None = None,
              min_count: int|float = 1, top_n: int|None = None,
              max_filter_size: tuple[int]|int|None = None, bounds: ndarray|None = None,
              ) -> ndarray:
    """
    Get initial seeds for the clustering algorithm. This is done by binning the
    `points` into a grid of size `bin_size`. This is useful to drastically
    reduce the number of centroids that need to be processed and it is likely
    that the points in the same bin will end up in the same cluster. Typically
    the `bin_size` is set to the bandwidth of the kernel used in the mean-shift
    algorithm.

    Providing weights for the points will cause the bins to be weighted by the
    sum of the weights of the points in the bin instead of the count of the
    points in the bin. This is useful to prioritize heavier weighted regions of
    the data.

    The number of bins can additionally be reduced with the `min_count`,
    `top_n`, and `max_filter_size` parameters. Reducing the number of seeds can
    greatly speed up the mean-shift algorithm, however, using these parameters
    may also remove some important points that become unique clusters.

     * `min_count` only considers bins that contain at least that many
       points or that much total weight
     * `top_n` only considers that many of the most populated/weighted bins
     * `max_filter_size` only considers bins that are local maxima within the
       given filter size (must be odd integers >1)

    If all are used, all must be satisfied for a bin to be returned (i.e. the
    bin must contain at least `min_count` points and be in the `top_n` most
    populated bins). The `max_filter_size` is applied first, so if it is used,
    there will be significantly more spread-out bins that are available for the
    `top_n` parameters to filter from. The `min_count` is based on the weighted
    count of the bins if weights are provided. In this case, the `min_count` is
    not necessarily an integer and it is possible to have a `min_count` of less
    than 1 (e.g. 0.5) to allow for bins that have some weight but not a full
    point.

    The `bounds` parameter can be used to limit the bins to a specific range.
    This useful to automatically remove complete outliers. Additionally,
    providing the `bounds` even if they eliminate no bins speeds up the process
    greatly by allowing for faster allocation of the grid to search (almost
    twice as fast).
    """
    # About 9x to 15x faster than the original method by using histogramdd() instead of unique().
    # This one is likely easier to port to C as well (except for maybe argpartition).
    # This one uses more memory though (but in C could reuse the counts array).
    hist, bin_size, mins = build_histogram(points, bin_size, weights, bounds)
    seeds = get_seeds_from_histogram(hist, min_count=min_count, top_n=top_n, max_filter_size=max_filter_size)
    return points2coords(seeds, bin_size, mins)


def build_histogram(points: ndarray, bin_size: float|Sequence[float]|ndarray,
                    weights: Sequence[float]|ndarray|None, bounds: ndarray|None
                    ) -> tuple[ndarray, ndarray, ndarray]:
    """
    Builds a histogram of the given points with the specified bin size.
    Optionally, weights and bounds can be provided to weight the histogram and
    limit the bins to a specific range, respectively.

    Providing bounds also speeds up the process greatly by allowing for faster
    allocation of the grid to search (almost twice as fast).
    """
    bin_size = np.asarray(bin_size)
    points = np.round(points / bin_size).astype(int)

    if bounds is None:
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        bins = [np.arange(mn, mx+1) for mn, mx in zip(mins, maxs)]
    else:
        bounds = np.asarray(bounds)
        if bounds.shape[1] != 2:
            raise ValueError("bounds must be a 2D array with shape (n, 2)")
        if bounds.shape[0] != points.shape[1]:
            raise ValueError("bounds must have the same number of rows as points has columns")
        bounds = np.round(bounds / np.atleast_2d(bin_size)).astype(int)
        mins = bounds[:, 0]
        bins = [np.arange(bound[0], bound[1]+1) for bound in bounds]

    hist, _ = np.histogramdd(points, bins=bins, weights=weights)
    return hist, bin_size, mins


def get_seeds_from_histogram(hist: ndarray, *, min_count: float = 1, top_n: int|None = None,
                             max_filter_size: tuple[int]|int|None = None) -> ndarray:
    """
    Get the seeds for the mean shift algorithm from the histogram of the data.
    See `get_seeds()` for more details on the parameters and how they affect
    the returned seeds.

    This is different from `get_seeds()` in that:
        * It takes a histogram instead of the raw points
        * It returns the seeds in histogram coordinates instead of the original
          coordinates (i.e. the seeds are integers that correspond to the bins
          of the histogram)
    """

    # Set every bin to 0 that has a higher max within max_filter_size
    if max_filter_size is not None:
        hist[hist < maximum_filter(hist, size=max_filter_size, mode='constant')] = 0

    if min_count <= 1 and top_n is None:
        seeds = np.argwhere(hist)
    elif top_n is None:
        seeds = np.argwhere(hist >= min_count)
    else:
        # We have to consider top-n points, semi-sort the data
        # Since we know that top_n is relatively small, in C we could instead try something a bit
        # more complicated to get the top_n points without a full partition.
        partition = np.argpartition(hist, -top_n, axis=None)
        if min_count > 1:
            # Adjust the min_count to be the minimum of the top_n most populated bins
            index = np.unravel_index(partition[-top_n], hist.shape)
            min_count = max(hist[index], min_count)
            seeds = np.argwhere(hist >= min_count)
        else:
            # Essentially the same as the case above and could probably just be done that way
            indices = np.unravel_index(partition[-top_n:], hist.shape)
            seeds = np.transpose(indices)

    return seeds


def points2coords(points: ndarray, bin_size: float|Sequence[float]|ndarray, mins: ndarray) -> ndarray:
    """
    Convert points in histogram coordinates to the original coordinates.

    Parameters:
    points : ndarray, shape (n, d)
        Points in histogram coordinates (i.e. integers that correspond to the bins of the histogram)
    bin_size : float or sequence of floats
        The size of the bins used to compute the histogram.
    mins : ndarray, shape (d,)
        The minimum values for each dimension that correspond to the first bin of the histogram.

    Returns:
    coords : ndarray, shape (n, d)
        Points in original coordinates.
    """
    return (points + 0.5 + mins) * np.asarray(bin_size)


def coords2points(coords: ndarray, bin_size: float|Sequence[float], mins: ndarray) -> ndarray:
    """
    Convert points in original coordinates to histogram coordinates.

    Parameters:
    coords : ndarray, shape (n, d)
        Points in original coordinates.
    bin_size : float or sequence of floats
        The size of the bins used to compute the histogram.
    mins : ndarray, shape (d,)
        The minimum values for each dimension that correspond to the first bin of the histogram.

    Returns:
    points : ndarray, shape (n, d)
        Points in histogram coordinates, not necessarily integers though
    """
    return coords / np.asarray(bin_size) - 0.5 - mins


##### KERNEL FUNCTIONS #####

def make_gaussian_kernel(bandwidth: float|ndarray,
                         weights: ndarray|None = None) -> Callable:
    """
    Make a Gaussian kernel function for mean-shift clustering.

    Parameters:
    bandwidth : float or ndarray
        The standard deviation of the Gaussian kernel. If a sequence, the
        length must match the number of dimensions.
    weights : ndarray, optional
        The weights for each sample. If not provided, all samples are
        weighted equally.

    Returns:
    gaussian_kernel : callable
        A function that computes the kernel for a given centroid and set of
        points. The function takes two arguments: a point (p) and a set of
        points (pts). It returns the kernel value for each point in pts with
        respect to p.
    """
    if weights is not None:
        weights = np.asarray(weights)
        #weights *= factor

    is_scalar = _is_scalar(bandwidth)
    bandwidth = np.asarray(bandwidth)
    if is_scalar:
        bandwidth = bandwidth.squeeze()
    exponent_factor = -0.5 / (bandwidth * bandwidth)

    if is_scalar:
        #factor = 1 / (bandwidth ** dim * (2 * np.pi) ** (dim/2))  # will cancel out anyways
        if weights is not None:
            def gaussian_kernel(p, pts):
                return weights * np.exp(euclidean_distance_2(p, pts)*exponent_factor)
        else:
            def gaussian_kernel(p, pts):
                return np.exp(euclidean_distance_2(p, pts)*exponent_factor)
    else:
        #factor = 1 / (bandwidth.prod() * (2 * np.pi) ** (dim/2))  # will cancel out anyways
        if weights is not None:
            def gaussian_kernel(p, pts):
                diffs = p - pts
                return weights * np.exp(np.sum(diffs*diffs*exponent_factor, axis=1))
        else:
            def gaussian_kernel(p, pts):
                diffs = p - pts
                return np.exp(np.sum(diffs*diffs*exponent_factor, axis=1))
    return gaussian_kernel


def _is_scalar(x: float|ndarray|Sequence[float]) -> bool:
    if np.isscalar(x):
        return True
    if isinstance(x, Sized) and len(x) == 1 and np.isscalar(x[0]):
        return True
    return False


def make_adaptive_gaussian_kernel(h_global: float|ndarray,
                                  pilot: ndarray, g: float,
                                  bin_size: float|ndarray, mins: ndarray,
                                  weights: ndarray|None = None,
                                  lambda_min: float = 0.2,
                                  lambda_max: float = 5.0) -> Callable:
    """
    Make a Gaussian kernel function for mean-shift clustering that uses
    Abramson-Silverman adaptive bandwidths at the start of the mean-shift
    algorithm. The bandwidths are not updated during the iterations of the
    mean-shift algorithm.

    Parameters:
    h_global : float or ndarray
        The global bandwidth, possibly different for each dimension. This is
        used as the base bandwidth that is scaled by the adaptive bandwidths.
        If a sequence, the length must match the number of dimensions.
    pilot : ndarray
        The pilot density estimate.
    g : float
        The smoothing parameter for the Abramson-Silverman adaptive bandwidths.
    bin_size : float or ndarray
        The size of each bin for the pilot density estimate. If a sequence, the
        length must match the number of dimensions.
    mins : ndarray
        The minimum values for each dimension.
    weights : ndarray, optional
        The weights for each sample. If not provided, all samples are
        weighted equally.

    Returns:
    gaussian_kernel : callable
        A function that computes the kernel for a given centroid and set of
        points. The function takes two arguments: a point (p) and a set of
        points (pts). It returns the kernel value for each point in pts with
        respect to p, using the adaptive bandwidths.
    """
    if g == 1.0 and pilot.sum() == 0:
        return make_gaussian_kernel(h_global, weights)

    if weights is not None:
        weights = np.asarray(weights)
    bin_size = np.asarray(bin_size)
    mins_0_5 = mins + 0.5

    is_scalar = _is_scalar(h_global)
    h_global = np.asarray(h_global)
    if is_scalar:
        h_global = h_global.squeeze()

    class _BandwidthCache(dict):
        def __getitem__(self, p):
            p_id = id(p)
            if p_id not in self:
                self[p_id] = retval = -0.5/_h_2(p, bin_size, mins_0_5, h_global, pilot, g, lambda_min, lambda_max)
            else:
                retval = super().__getitem__(p_id)
            return retval
    bandwidths = _BandwidthCache()

    if is_scalar:
        if weights is not None:
            def adaptive_gaussian_kernel(p, pts):
                return weights * np.exp(euclidean_distance_2(p, pts)*bandwidths[p])
        else:
            def adaptive_gaussian_kernel(p, pts):
                return np.exp(euclidean_distance_2(p, pts)*bandwidths[p])
    else:
        if weights is not None:
            def adaptive_gaussian_kernel(p, pts):
                diffs = p - pts
                return weights * np.exp(np.sum(diffs*diffs*bandwidths[p], axis=1))
        else:
            def adaptive_gaussian_kernel(p, pts):
                diffs = p - pts
                return np.exp(np.sum(diffs*diffs*bandwidths[p], axis=1))

    return adaptive_gaussian_kernel
        

def make_dynamic_gaussian_kernel(h_global: float|ndarray,
                                 pilot: ndarray, g: float,
                                 bin_size: float|ndarray, mins: ndarray,
                                 weights: ndarray|None = None,
                                 lambda_min: float = 0.2,
                                 lambda_max: float = 5.0) -> Callable:
    """
    Make a Gaussian kernel function for mean-shift clustering that uses
    Abramson-Silverman adaptive bandwidths during every iteration of the
    mean-shift algorithm.

    Parameters:
    h_global : float or ndarray
        The global bandwidth, possibly different for each dimension. This is
        used as the base bandwidth that is scaled by the adaptive bandwidths.
        If a sequence, the length must match the number of dimensions.
    pilot : ndarray
        The pilot density estimate.
    g : float
        The smoothing parameter for the Abramson-Silverman adaptive bandwidths.
    bin_size : float or ndarray
        The size of each bin for the pilot density estimate. If a sequence, the
        length must match the number of dimensions.
    mins : ndarray
        The minimum values for each dimension.
    weights : ndarray, optional
        The weights for each sample. If not provided, all samples are
        weighted equally.

    Returns:
    gaussian_kernel : callable
        A function that computes the kernel for a given centroid and set of
        points. The function takes two arguments: a point (p) and a set of
        points (pts). It returns the kernel value for each point in pts with
        respect to p, using the adaptive bandwidths.
    """
    if g == 1.0 and pilot.sum() == 0:
        return make_gaussian_kernel(h_global, weights)

    if weights is not None:
        weights = np.asarray(weights)
    bin_size = np.asarray(bin_size)
    mins_0_5 = mins + 0.5

    is_scalar = _is_scalar(h_global)
    h_global = np.asarray(h_global)

    if is_scalar:
        h_global = h_global.squeeze()
        if weights is not None:
            def dynamic_gaussian_kernel(p, pts):
                return weights * np.exp(euclidean_distance_2(p, pts)*-0.5/_h_2(p, bin_size, mins_0_5, h_global, pilot, g, lambda_min, lambda_max))
        else:
            def dynamic_gaussian_kernel(p, pts):
                return np.exp(euclidean_distance_2(p, pts)*-0.5/_h_2(p, bin_size, mins_0_5, h_global, pilot, g, lambda_min, lambda_max))
    else:
        if weights is not None:
            def dynamic_gaussian_kernel(p, pts):
                diffs = p - pts
                return weights * np.exp(np.sum(diffs*diffs*-0.5/_h_2(p, bin_size, mins_0_5, h_global, pilot, g, lambda_min, lambda_max), axis=1))
        else:
            def dynamic_gaussian_kernel(p, pts):
                diffs = p - pts
                return np.exp(np.sum(diffs*diffs*-0.5/_h_2(p, bin_size, mins_0_5, h_global, pilot, g, lambda_min, lambda_max), axis=1))

    return dynamic_gaussian_kernel


def _h_2(p, bin_size, mins_0_5, h_global, pilot, g, lambda_min: float = 0.2, lambda_max: float = 5.0):
    f_seed = _bilinear_interp1(pilot, p / bin_size - mins_0_5)
    h = h_global * np.sqrt(g / f_seed).clip(lambda_min, lambda_max)
    return h * h


##### ABRAMSON-SILVERMAN ADAPTIVE BANDWIDTHS #####

def estimate_bandwidth(
        points: ndarray,
        weights: ndarray|None = None,
        factor: float = 2.0,
        ) -> tuple[ndarray, float]:
    """
    Estimate the bandwidth to use with the mean shift algorithm using
    Silverman's rule of thumb. There are several assumptions that go into this
    method, mainly that the density of the data is normally distributed.

    From B. W. Silverman (1986), Density Estimation for Statistics and Data
    Analysis, Chapman & Hall/CRC, London.

    The univariate equation is on page 48, eqn (3.31).

    See also Scott's Multivariate Density Estimation (1992) and Wand & Jones,
    Kernel Smoothing (1995, Chapman & Hall).

    Parameters:
    points : ndarray, shape (n_samples, n_features)
        The data points for which to estimate the bandwidth.
    weights : ndarray, shape (n_samples,), optional
        The weights for each sample. If not provided, all samples are
        weighted equally.
    factor : float, optional
        The factor to use in Silverman's rule of thumb. Default is 2.0. A value
        of 4.0 results in the standard "1.06" factor for univariate data that
        matches the normal calculation. Using 2.0 results in the "0.9" factor
        which Silverman later recommended.
    
    Returns:
    bandwidth : ndarray, shape (n_features,)
        The estimated bandwidth for each dimension.
    silverman_factor : float
        The factor used in the calculation of the bandwidth.
    """
    n, d = points.shape
    if weights is not None:
        weights = np.asarray(weights)
        wsum = weights.sum(axis=0)
        n_eff = wsum * wsum / (weights * weights).sum(axis=0)  # effective sample size
        avg = np.average(points, axis=0, weights=weights)
        diff = points - avg
        stddevs = np.sqrt(np.average(diff * diff, axis=0, weights=weights) * (n / (n - 1)))  # assumes all weights are positive
    else:
        n_eff = n
        stddevs = np.std(points, axis=0, ddof=1)
    q25s, q75s = np.quantile(points, [.25, .75], axis=0, weights=weights, method="inverted_cdf")
    iqrs = q75s - q25s
    A = np.where(iqrs > 0, np.minimum(stddevs, iqrs / 1.34), stddevs)
    silverman_factor = (factor / (n_eff * (d + 2))) ** (1.0 / (d + 4))
    return silverman_factor * A, silverman_factor


def bandwidths_to_iso(silverman_factor: float, bandwidths: ndarray) -> float:
    """
    Convert a set of bandwidths to a single isotropic bandwidth using the geometric mean. This is
    useful for the case where the bandwidths are different for each dimension, but we want to use
    a single bandwidth for all dimensions.
    """
    return np.exp(np.mean(np.log(bandwidths / silverman_factor))) * silverman_factor


def compute_pilot_density(hist: ndarray, bin_size: float|Sequence[float]|ndarray,
                          density_floor_factor: float=0.75) -> tuple[ndarray, float]:
    """
    Compute the pilot density for Abramson-Silverman adaptive bandwidths. The
    pilot density is computed by normalizing the histogram of the data.

    Parameters:
    hist : ndarray
        The histogram of the data, typically computed using np.histogramdd()
        with the same bin size as the one used for the seeds in the mean shift
        algorithm. Can be weighted by providing weights to the histogram
        function, which will prioritize heavier weighted regions of the data.
    bin_size : float or sequence of floats
        The size of the bins used to compute the histogram.
    density_floor_factor : float, optional
        The factor by which to multiply the geometric mean to set the floor for
        the pilot density. Default is 0.75.

    Returns:
    pilot : ndarray
        The computed pilot density.
    g : float
        The geometric mean of the pilot density.
    """
    total_weight = hist.sum()
    if total_weight == 0:
        return np.zeros(hist.shape), 1.0

    # Pilot density (normalized so it integrates to ~1)
    bin_area = (bin_size ** hist.ndim) if isinstance(bin_size, (int, float)) else np.prod(bin_size)
    pilot = hist / (total_weight * bin_area)

    # Weighted geometric mean over non-empty bins
    nonzero = hist > 0
    # geometric mean - exp(mean(log(x))) is the same as prod(x)
    g = np.exp(np.sum(np.log(pilot[nonzero]) * hist[nonzero]) / total_weight)

    # Apply a floor
    pilot = np.maximum(pilot, g * density_floor_factor)
    return pilot, g

def _abramson_silverman(f_seeds: ndarray, h_global: float|ndarray, g: float, lambda_min: float, lambda_max: float) -> ndarray:
    # Abramson's square-root law
    # To account for divide-by-zero: (but the flooring of the pilot density should prevent this)
    # lambda_i = np.where(f_seeds > 0, np.sqrt(g / f_seeds), 1.0).clip(lambda_min, lambda_max)
    lambda_i = np.sqrt(g / f_seeds).clip(lambda_min, lambda_max)
    return h_global * lambda_i  # TODO: if h_global is an array, we need to do some broadcasting?


def abramson_silverman_grid(points: ndarray, h_global: float,
                            pilot: ndarray, g: float,
                            lambda_min: float = 0.2, lambda_max: float = 5.0) -> ndarray:
    """
    Compute the Abramson-Silverman adaptive bandwidths for the given points.
    This method uses the pilot density to compute the bandwidth for each point
    based on the square-root law. The bandwidth for each point is given by:
        h_i = h_global * sqrt(g / f(p_i))
    where h_global is the global bandwidth, g is the geometric mean of the
    pilot density, and f(p_i) is the pilot density at the point p_i. The
    bandwidths are clipped to be within the range
    [lambda_min * h_global, lambda_max * h_global].

    Parameters:
    points : ndarray, shape (n_samples, n_features)
        The points for which to compute the adaptive bandwidths. These must be
        in the same space as the pilot density (i.e. binned).
    h_global : float or ndarray, shape (n_features,)
        The global bandwidth, possibly different for each dimension.
    pilot : ndarray
        The pilot density. See `compute_pilot_density()`. Must not contain any
        non-positive values.
    g : float
        The geometric mean of the pilot density. See `compute_pilot_density()`.
    lambda_min : float, optional
        The minimum scaling factor for the adaptive bandwidths. Default is 0.2.
    lambda_max : float, optional
        The maximum scaling factor for the adaptive bandwidths. Default is 5.0.

    Returns:
    bandwidths : ndarray, shape (n_samples,) or (n_samples, n_features)
        The computed adaptive bandwidths for each point.
    """
    if points.shape[1] != pilot.ndim:
        raise ValueError("points must have the same number of dimensions as the pilot density")
    if np.any(points < 0) or np.any(points >= np.array(pilot.shape)):
        raise ValueError("points must be within the bounds of the pilot density")
    if points.dtype.kind not in 'iu':
        raise ValueError("points must be integer indices corresponding to the bins of the pilot density")
    if g == 1.0 and pilot.sum() == 0:
        return np.full(len(points), h_global)

    f_seeds = pilot[tuple(points.T)]  # e.g. pilot[points[:, 0], points[:, 1]] for 2D data
    return _abramson_silverman(f_seeds, h_global, g, lambda_min, lambda_max)


def _bilinear_interp(data: ndarray, points: ndarray) -> ndarray:
    """
    Perform bilinear interpolation on the data at the given points.

    Parameters:
    data : ndarray, shape (nx, ny)
        The 2D array of data to interpolate from.
    points : ndarray, shape (n_samples, 2)
        The points at which to perform the interpolation. The first column is
        the x-coordinate and the second column is the y-coordinate.

    Returns:
    interpolated_values : ndarray, shape (n_samples,)
        The interpolated values at the given points.
    """
    # TODO: support n-dimensional data
    px = points[:, 0]
    py = points[:, 1]
    nx, ny = data.shape
    x0 = np.clip(np.floor(px).astype(int), 0, nx - 1)
    x1 = np.clip(np.ceil(px).astype(int), 0, nx - 1)
    y0 = np.clip(np.floor(py).astype(int), 0, ny - 1)
    y1 = np.clip(np.ceil(py).astype(int), 0, ny - 1)
    x_frac = px - x0
    y_frac = py - y0
    return (data[x0, y0] * (1 - x_frac) * (1 - y_frac) + data[x1, y0] * x_frac * (1 - y_frac) +
            data[x0, y1] * (1 - x_frac) * y_frac       + data[x1, y1] * x_frac * y_frac)


def _bilinear_interp1(data: ndarray, point: ndarray) -> float:
    """
    Perform bilinear interpolation on the data at the given point.

    Parameters:
    data : ndarray, shape (nx, ny)
        The 2D array of data to interpolate from.
    point : ndarray, shape (2,)
        The point at which to perform the interpolation. The first element is
        the x-coordinate and the second element is the y-coordinate.

    Returns:
    interpolated_value : float
        The interpolated value at the given point.
    """
    px, py = point
    nx, ny = data.shape
    x0 = np.clip(np.floor(px).astype(int), 0, nx - 1)
    x1 = np.clip(np.ceil(px).astype(int), 0, nx - 1)
    y0 = np.clip(np.floor(py).astype(int), 0, ny - 1)
    y1 = np.clip(np.ceil(py).astype(int), 0, ny - 1)
    x_frac = px - x0
    y_frac = py - y0
    return (data[x0, y0] * (1 - x_frac) * (1 - y_frac) + data[x1, y0] * x_frac * (1 - y_frac) +
            data[x0, y1] * (1 - x_frac) * y_frac       + data[x1, y1] * x_frac * y_frac)


def abramson_silverman_interp(coords: ndarray, bin_size: float|Sequence[float], mins: ndarray,
                              h_global: float|ndarray, pilot: ndarray, g: float,
                              lambda_min: float = 0.2, lambda_max: float = 5.0):
    """
    Mostly the same as `abramson_silverman_grid()`, but it allows 
    interpolation of the pilot density at the given coordinates instead of just
    using the binned values. This is useful to get smoother bandwidths and to
    allow for points that are not exactly at the bin centers. Unlike the grid
    version, this version uses coordinates in the original space, not the
    binned space (the `bin_size` and `mins` are used to convert).

    Parameters:
    coords : ndarray, shape (n_samples, n_features)
        The coordinates for which to compute the adaptive bandwidths. These are
        in the original space, not the binned space.
    bin_size : float or sequence of floats
        The size of the bins used to compute the histogram.
    mins : ndarray, shape (n_features,)
        The minimum values for each dimension that correspond to the first bin
        of the histogram.
    h_global : float or ndarray, shape (n_features,)
        The global bandwidth. Possibly different for each dimension.
    pilot : ndarray
        The pilot density. See `compute_pilot_density()`. Must not contain any
        non-positive values.
    g : float
        The geometric mean of the pilot density. See `compute_pilot_density()`.
    lambda_min : float, optional
        The minimum scaling factor for the adaptive bandwidths. Default is 0.2.
    lambda_max : float, optional
        The maximum scaling factor for the adaptive bandwidths. Default is 5.0.
    
    Returns:
    bandwidths : ndarray, shape (n_samples,) or (n_samples, n_features)
        The computed adaptive bandwidths for each coordinate.
    """
    if g == 1.0 and pilot.sum() == 0:
        return np.full(len(coords), h_global)
    f_seeds = _bilinear_interp(pilot, coords2points(coords, bin_size, mins))
    return _abramson_silverman(f_seeds, h_global, g, lambda_min, lambda_max)
