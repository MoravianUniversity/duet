"""
Weighted Mean Shift

From https://www.sciencedirect.com/science/article/abs/pii/S0165168412000722
"""

from typing import Sequence, Callable

import numpy as np
from numpy import ndarray

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
        the number of initial seeds if some centroids are close to each other.
    """
    if kernel is None: kernel = make_gaussian_kernel(bandwidth)
    centroids = get_seeds(points, bandwidth) if seeds is None else np.array(seeds)

    # squared tolerance so we don't have to take the square root
    tol_2 = bandwidth * bandwidth * convergence_tol * convergence_tol

    for i, centroid in enumerate(centroids):  # can be parallelized
        dist_2 = np.inf
        while dist_2 > tol_2:
            # Shift the centroid
            weights = kernel(centroid, points)
            p_new = (weights[:, None] * points).sum(0) / weights.sum()

            # Check if the point has converged
            dist_2 = euclidean_distance_2(p_new, centroid)

            # Update the centroid
            centroids[i] = p_new

    # Combine all centroids that are close to each other
    return remove_near_duplicates(centroids, bandwidth)


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


def get_seeds(points: ndarray, bin_size,
              min_count: int = 1, top_n: int|None = None, bounds: ndarray|None = None
              ) -> ndarray:
    """
    Get initial seeds for the clustering algorithm. This is done by binning the
    `points` into a grid of size `bin_size`. This is useful to drastically
    reduce the number of centroids that need to be processed and it is likely
    that the points in the same bin will end up in the same cluster. Typically
    the `bin_size` is set to the bandwidth of the kernel used in the mean-shift
    algorithm.

    The number of bins can additionally be reduced with the `min_count` and
    `top_n` parameters. Reducing the number of seeds can greatly speed up the
    mean-shift algorithm, however, using these parameters may also remove some
    important points that become unique clusters.

     * `min_count` only considers bins that contain at least that many points
     * `top_n` only considers that many of the most populated bins

    If both are used, both must be satisfied for a bin to be returned (i.e. the
    bin must contain at least `min_count` points and be in the `top_n` most
    populated bins).

    The bounds parameter can be used to limit the bins to a specific range. This
    useful to automatically remove complete outliers. Additionally, providing
    the bounds even if they eliminate no bins speeds up the process greatly by
    allowing for faster allocation of the grid to search (almost twice as fast).
    """
    # TODO: support weights for the points?
    # TODO: this uses bin left edges, but maybe bin centers would be better?
    bin_size = np.asarray(bin_size)
    points = np.round(points / bin_size).astype(int)

    # New Method
    # About 9x to 15x faster than the original method by using histogramdd() instead of unique().
    # This one is likely easier to port to C as well (except for maybe argpartition).
    # This one uses more memory though (but in C could reuse the counts array).
    if bounds is None:
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        bins = [np.arange(mn, mx+1) for mn, mx in zip(mins, maxs)]
    else:
        bounds = np.asarray(bounds)
        if bounds.shape[1] != 2:
            raise ValueError("bounds must be a 2D array with shape (n, 2)")
        if bounds.shape[0] != points.shape[1]:
            raise ValueError("bounds must have the same number of rows as points")
        bounds = np.round(bounds / bin_size).astype(int)
        mins = bounds[:, 0]
        bins = [np.arange(bound[0], bound[1]+1) for bound in bounds]

    counts, _ = np.histogramdd(points, bins=bins)
    if min_count <= 1 and top_n is None:
        points = np.argwhere(counts)
    elif top_n is None:
        points = np.argwhere(counts >= min_count)
    else:
        # We have to consider top-n points, semi-sort the data
        # Since we know that top_n is relatively small, in C we could instead try something a bit
        # more complicated to get the top_n points without a full partition.
        partition = np.argpartition(counts, -top_n, axis=None)
        if min_count > 1:
            # Adjust the min_count to be the minimum of the top_n most populated bins
            index = np.unravel_index(partition[-top_n], counts.shape)
            min_count = max(counts[index], min_count)
            points = np.argwhere(counts >= min_count)
        else:
            # Essentially the same as the case above and could probably just be done that way
            indices = np.unravel_index(partition[-top_n:], counts.shape)
            points = np.transpose(indices)

    return (points + mins) * bin_size

    # # Original Method
    # # Uses numpy's unique function which is the time limiting step (~99% of the time).
    # # Tried using sets/dicts but they were about 25% slower.
    # # Never added bounds support to this one.
    # if min_count <= 1 and top_n is None:
    #     return np.unique(points, axis=0) * bin_size
    #     # return np.array({tuple(p) for p in points}) * bin_size

    # seeds, bin_counts = np.unique(points, axis=0, return_counts=True)
    # # from collections import defaultdict
    # # bin_counts = defaultdict(int)
    # # for p in points: bin_counts[tuple(p)] += 1

    # if top_n is None:
    #     ind = bin_counts > min_count
    # else:
    #     ind = np.argpartition(bin_counts, -top_n)[-top_n:]
    #     if min_count > 1:
    #         ind = ind[bin_counts[ind] >= min_count]
    # # if top_n is not None:
    # #     min_count = max(sorted(bin_counts.values(), reverse=True)[top_n], min_count)
    # # points = np.array([p for p, c in bin_counts.items() if c >= min_count])

    # return seeds[ind] * bin_size
    # # return points * bin_size


##### KERNEL FUNCTION #####

def make_gaussian_kernel(bandwidth: float|Sequence[float],
                         weights: Sequence[float]|None = None) -> Callable:
    """
    Make a Gaussian kernel function for mean-shift clustering.

    Parameters:
    bandwidth : float or sequence of floats
        The standard deviation of the Gaussian kernel. If a sequence, the
        length must match the number of dimensions.
    weights : sequence of floats, optional
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

    if np.isscalar(bandwidth) or len(bandwidth) == 1 and np.isscalar(bandwidth[0]):
        bandwidth = np.asarray(bandwidth).squeeze()
        #factor = 1 / (bandwidth ** dim * (2 * np.pi) ** (dim/2))  # will cancel out anyways
        exponent_factor = -0.5 / (bandwidth * bandwidth)
        if weights is not None:
            def gaussian_kernel(p, pts):
                return weights * np.exp(euclidean_distance_2(p, pts)*exponent_factor)
        else:
            def gaussian_kernel(p, pts):
                return np.exp(euclidean_distance_2(p, pts)*exponent_factor)
    else:
        bandwidth = np.asarray(bandwidth)
        #factor = 1 / (bandwidth.prod() * (2 * np.pi) ** (dim/2))  # will cancel out anyways
        exponent_factor = -0.5 / (bandwidth * bandwidth)
        if weights is not None:
            def gaussian_kernel(p, pts):
                diffs = p - pts
                return weights * np.exp(np.sum(diffs*diffs*exponent_factor, axis=1))
        else:
            def gaussian_kernel(p, pts):
                diffs = p - pts
                return np.exp(np.sum(diffs*diffs*exponent_factor, axis=1))
    return gaussian_kernel
