#pragma once
#include <vector>
#include <array>
#include <type_traits>

#include "fast_math.hpp"
#include "constexpr_array.hpp"
#include "largest_k.h"


/**
 * Mean Shift clustering algorithm parameters. Define a struct that inherits
 * from this struct and pass it as the template parameter to `MeanShift`. This
 * allows you to customize the behavior of the mean-shift algorithm at compile
 * time.
 */
struct MeanShiftParams {
    /** Number of dimensions in the points being clustered with mean-shift. */
    static constexpr unsigned int dim = 2;

    /**
     * The bandwidth of the Gaussian kernel used in the mean-shift algorithm.
     * Also used as the bin size in the histogram. This can be a single float
     * or an array of floats. Using a single float will result in the same
     * bandwidth for all dimensions and be slightly faster.
     */
    static constexpr float bandwidth = 0.2f;
    //static constexpr std::array<float, dim> bandwidth = {0.2f, 0.2f};

    /**
     * Convergence tolerance for the mean shift algorithm. This is the relative
     * distance between the current and new centroid for the algorithm to
     * consider it converged (scaled by the bandwidth).
     */
    static constexpr float convergence_tol = 0.1f;

    /**
     * The minimum bounds of the histogram and grid filtering.
     * These are used to limit the bins to a specific range, which speeds up
     * the mean-shift algorithm by eliminating outliers that would take the
     * longest to converge.
     * Any value outside of these bounds will be ignored in the histogram.
     */
    static constexpr std::array<float, dim> min_bounds = {-1.0f, -1.0f};

    /**
     * The maximum bounds of the histogram and grid filtering.
     * These are used to limit the bins to a specific range, which speeds up
     * the mean-shift algorithm by eliminating outliers that would take the
     * longest to converge.
     * Any value outside of these bounds will be ignored in the histogram.
     */
    static constexpr std::array<float, dim> max_bounds = {1.0f, 1.0f};

    /**
     * If true, assume that all points are within the bounds of the histogram
     * (slight speedup). This effects seed generation and effects mean-shift if
     * grid filtering is enabled. If this is true and a point is out of bounds,
     * it will generate undefined behavior, possibly crashing.
     */
    static constexpr bool assume_in_bounds = true;

    /**
     * If true, use grid-filtering to speed up the mean-shift algorithm but may
     * result in slightly worse results while resulting in a significant
     * speedup. If this is enabled, at most 255 centroids/seeds can be provided
     * to `mean_shift()`.
     */
    static constexpr bool grid_filtering = true;

    /** Minimum count of points in a bin to consider it a seed, >= 1 */
    static constexpr int min_count = 3;

    /** Number of most populated bins to consider as seeds, >= 0, 0 to disable */
    static constexpr int top_n = 20;
};


/**
 * Mean Shift clustering algorithm.
 * This is a template class that takes a `MeanShiftParams` struct as a template
 * parameter. The struct defines the compile-time parameters for the algorithm.
 */
template<typename Params = MeanShiftParams>
class MeanShift {
public:
    typedef MeanShift<Params> self_t;

    /** Number of dimensions in the points being clustered */
    static constexpr int dim = Params::dim;
    static_assert(dim > 0, "`dim` must be greater than 0");

    /** N-dimensional point type, used in mean-shift clustering */
    typedef std::array<float, Params::dim> point_t;

    /**
     * Bandwidth of the Gaussian kernel used in the mean-shift algorithm,
     * either a float or an array of floats
     */
    static constexpr auto bandwidth = Params::bandwidth;

    /** Convergence tolerance for the mean shift algorithm. */
    static constexpr float convergence_tol = Params::convergence_tol;
    static constexpr float convergence_tol_2 = convergence_tol * convergence_tol;

    /** Minimum bounds of the histogram and grid filtering */
    static constexpr point_t min_bounds = Params::min_bounds;

    /** Maximum bounds of the histogram and grid filtering */
    static constexpr point_t max_bounds = Params::max_bounds;

    /** If true, assume that all points are within the bounds of the histogram */
    static constexpr bool assume_in_bounds = Params::assume_in_bounds;

    /** If true, use grid-filtering to speed up the mean-shift algorithm */
    static constexpr bool grid_filtering = Params::grid_filtering;

    /** Minimum count of points in a bin to consider it a seed, >= 1 */
    static constexpr int min_count = Params::min_count;
    static_assert(min_count >= 1, "`min_count` must be at least 1");

    /** Number of most populated bins to consider as seeds, >= 0, 0 to disable */
    static constexpr int top_n = Params::top_n;
    static_assert(top_n >= 0, "`top_n` must be at least 0");

private:
    typedef typename std::remove_const<decltype(bandwidth)>::type bandwidth_t;
    constexpr static bool is_single_bandwidth = std::is_same<bandwidth_t, float>::value;

    // Derive a scalar or an array
    template <typename T, typename F>
    static constexpr typename std::enable_if<std::is_same<T, float>::value, float>::type __make(const T& value, F func) { return func(value); }
    template <typename T, typename F>
    static constexpr typename std::enable_if<std::is_same<T, point_t>::value, point_t>::type __make(const T& value, F func) { return make_transformed_array(value, func); }

    // Get the value from a scalar or an array
    template <typename T>
    static constexpr typename std::enable_if<std::is_same<T, float>::value, float>::type _get(const T& value, int i) { return value; }
    template <typename T>
    static constexpr typename std::enable_if<!std::is_same<T, float>::value, float>::type _get(const T& value, int i) { return value[i]; }

    // Derived constants
    static constexpr float __make_exp_factor(float x) { return -0.5f / (x * x); }
    static constexpr float __make_bin_size_inv(float x) { return 1.0f / x; }
    static constexpr bandwidth_t exponent_factor = __make(bandwidth, __make_exp_factor);
    static constexpr bandwidth_t bin_size_inv = __make(bandwidth, __make_bin_size_inv);

    static constexpr int16_t __make_hist_shape(size_t i) {
        return (int16_t)(max_bounds[i] * _get(bin_size_inv, i) + 0.5f) - 
            (int16_t)(min_bounds[i] * _get(bin_size_inv, i) + 0.5f) + 1;
    }
    static constexpr int __make_hist_size(const std::array<int16_t, dim>& shape) {
        int size = 1;
        for (int i = 0; i < dim; ++i) { size *= shape[i]; }
        return size;
    }

    // The shape and size of the histogram computed from the bounds and bin size
    static constexpr std::array<int16_t, dim> hist_shape = make_array<int16_t, dim>(__make_hist_shape);
    static constexpr int hist_size = __make_hist_size(hist_shape);

public:
    MeanShift() = default;

    /**
     * Get initial seeds for the clustering algorithm. This is done by binning the
     * `points` into a grid. This drastically reduces the number of centroids that
     * need to be processed without majorly affecting quality since points in the
     * same bin will end up in the same cluster.
     *
     * Template parameters:
     * - `bandwidth`: the size of each bin in the histogram, set to the bandwidth
     *   of the kernel used in the mean-shift algorithm.
     * - `min_bounds` and `max_bounds`: the bounds of the histogram, used to limit
     *   the bins to a specific range. By eliminating these outliers, mean-shift
     *   speeds up greatly since it removes the points that would take the longest
     *   to time to converge.
     * - `min_count`: the minimum number of points in a bin to consider it a seed.
     * - `top_n`: the number of most populated bins to consider as seeds.
     *
     * Both `min_count` and `top_n` constants greatly speed up the mean-shift
     * algorithm, but they may also remove some important points that become unique
     * clusters. If both are used, both must be satisfied for a bin to be included.
     * (i.e. the bin must contain at least `min_count` points and be in the `top_n`
     * most populated bins).
     */
    void compute_seeds(
        const std::vector<point_t>& points,
        std::vector<point_t>& seeds
    ) {
        // TODO: support weights for the points?
        // TODO: this uses bin left edges, but maybe bin centers would be better?

        // Construct the histogram of the points
        int counts[hist_size];
        hist_nd(points, counts);

        int min_count = self_t::min_count;  // default minimum count of points in a bin to consider it a seed
        if (top_n >= 1) {
            // Adjust the min_count to be the minimum of the top_n most populated bins
            // NOTE: this works differently than the Python version slightly, if there are ties for
            // target value this will return all of them and thus can return more than `TOP_N` bins.
            int target = largest_k(counts, hist_size, top_n);
            min_count = target > min_count ? target : min_count;
        }

        // Find the points that have at least min_count in the histogram
        for (int i = 0; i < hist_size; i++) {
            if (counts[i] >= min_count) {
                // Convert the linear index back to coordinates
                seeds.emplace_back();
                index_to_point(i, seeds.back());
            }
        }
    }

    /**
     * Perform weighted mean shift clustering on the given points using a Gaussian
     * kernel.
     * 
     * Parameters:
     *  - points : The input data
     *  - weights : The weights for each point, used to determine the influence of
     *    each point on the clustering. This is used to give more/less importance
     *    to each point.
     *  - centroids : The initial centroids to start the clustering from. These are
     *    usually the output of `compute_seeds()`. The number of centroids should
     *    be significantly less than the number points. The resulting centroids will
     *    be written back to this array. There may be at most 255 centroids.
     *
     * Numerous template parameters may need to be tweaked.
     */
    void mean_shift(
        const std::vector<point_t>& points,
        const std::vector<float>& weights,
        std::vector<point_t>& centroids
    ) {
        int n = centroids.size();
        bool mask[n];
        memset(mask, 0, n * sizeof(bool));

        // TODO: use a finer grid than the histogram to increase accuracy
        std::conditional_t<grid_filtering, uint8_t[hist_size], uint8_t> grid;
        if (grid_filtering) {
            assert(n <= 255);  // grid filtering only works with at most 255 centroids
            memset(grid, 0xFF, sizeof(grid));
        }

        for (int c = 0; c < n; c++) {
            point_t& centroid = centroids[c];
            bool not_converged;
            do {
                if (grid_filtering) {
                    int grid_index = point_to_index(centroid);
                    if (assume_in_bounds || grid_index >= 0) { // no grid-filtering if the point is out of bounds, wait until the point is in bounds
                        if (grid[grid_index] == 0xFF) {
                            grid[grid_index] = c;
                        } else if (grid[grid_index] != c) {
                            // This centroid is in a bin that has already been visited by another centroid
                            mask[c] = true;  // mark this centroid as skipped
                            break;
                        }
                    }
                }

                float w_sum = 0.0f, pt_weight = 0.0f;
                point_t pt_new = {0.0f};

                for (int i = 0; i < points.size(); i++) {
                    const point_t& pt = points[i];

                    // Compute the Gaussian kernel weight for the current centroid and point
                    float exponent = 0.0f;
                    for (int d = 0; d < dim; d++) {
                        float diff = centroid[d] - pt[d];
                        // skip points that are too far away (2.5 std devs)
                        if (fabsf(diff) > 2.5f*_get(bandwidth, d)) { goto outer_continue; }
                        exponent += is_single_bandwidth ? diff * diff : _get(exponent_factor, d) * diff * diff;
                    }
                    if (is_single_bandwidth) { exponent *= _get(exponent_factor, 0); }
                    pt_weight = weights[i] * exp_fast_o1(exponent);
                    w_sum += pt_weight;

                    // Shift the centroid to the weighted average of the points
                    for (int d = 0; d < dim; d++) { pt_new[d] += pt_weight * pt[d]; }

                    outer_continue:;
                }

                // Normalization factor (sum of weights)
                float w_sum_inv = recip(w_sum);
                for (int d = 0; d < dim; d++) { pt_new[d] *= w_sum_inv; }

                // Check if the point has converged
                not_converged = !is_nearby(pt_new, centroid);

                // Update the centroid
                centroid = pt_new;
            } while (not_converged);
        }


        if (grid_filtering) {
            int p = 0;
            for (int c = 0; c < n; c++) {
                if (mask[c]) { continue; }  // skip points that have been filtered out
                if (c != p) { centroids[p] = centroids[c]; }
                p++;
            }
            centroids.resize(p);
        } else {
            // Combine all centroids that are close to each other
            remove_near_duplicates(centroids, mask);
        }
    }

private:
    /**
     * Remove points that are within a certain tolerance of each other.
     * 
     * The passed masks are used to skip points that should be ignored when
     * true. It is also used during processing and all points will be masked
     * once this function returns.
     */
    static void remove_near_duplicates(std::vector<point_t>& points, bool* mask) {
        int m = 0;
        for (int i = 0; i < points.size(); i++) {
            if (mask[i]) { continue; }  // skip points that have already been processed
            mask[i] = true;

            const point_t& pt = points[i];
            point_t temp_pt = pt;

            // find all nearby points and merge them
            int count = 1;  // number of nearby points merged
            for (int j = i+1; j < points.size(); j++) {
                if (mask[j]) { continue; }
                const point_t& pt_j = points[j];
                if (is_nearby(pt, pt_j)) {
                    // merge points i and j
                    mask[j] = true;
                    count += 1;
                    for (int k = 0; k < dim; k++) { temp_pt[k] += pt_j[k]; }
                }
            }

            // average the merged points
            if (count > 1) {
                float count_inv = recip(count);
                for (int j = 0; j < dim; j++) { temp_pt[j] *= count_inv; }
            }
            points[m++] = temp_pt;
        }
        points.resize(m);
    }

    /**
     * Convert a point in n-dimensional space to a linear index.
     * Returns -1 if the point is out of bounds.
     */
    static int point_to_index(
        const point_t& point,
        const std::array<int16_t, dim>& shape = hist_shape,
        const point_t& min_bounds = self_t::min_bounds
    ) {
        int index = 0;
        for (int d = 0; d < dim; d++) {
            // Convert the the point value to an integer coordinate in the bin grid
            int x = (int)round((point[d] - min_bounds[d]) * _get(bin_size_inv, d));
            // Check if the point is out of bounds
            if (assume_in_bounds) { assert(x >= 0 && x < shape[d]); }
            else if (x < 0 || x >= shape[d]) { return -1; }
            // Convert the integer coordinate to a linear index in the histogram
            index = index * shape[d] + x;
        }
        return index;
    }

    /**
     * Convert a linear index to a point in n-dimensional space.
     */
    static void index_to_point(
        int index, point_t& point,
        const std::array<int16_t, dim>& shape = hist_shape,
        const point_t& min_bounds = self_t::min_bounds
    ) {
        for (int d = dim - 1; d > 0; d--) {
            point[d] = (index % shape[d]) * _get(bandwidth, d) + min_bounds[d];
            index /= shape[d];
        }
        point[0] = index * _get(bandwidth, 0) + min_bounds[0];
    }

    /**
     * Calculate the histogram of points in n-dimensional space.
     */
    static void hist_nd(
        const std::vector<point_t>& points,
        int* counts                 // out, shape HIST_SHAPE (total size HIST_SIZE)
    ) {
        memset(counts, 0, hist_size * sizeof(int));  // TODO: use dsps_memset()
        for (const point_t & pt : points) {
            int index = point_to_index(pt);
            // If the point is in bounds, increment the count in the histogram
            if (assume_in_bounds || index >= 0) { counts[index] += 1; }
        }
    }

    /**
     * Check if two points are nearby within a tolerance. This computes a
     * weighted Euclidean distance between the points.
     */
    static bool is_nearby(const point_t& a, const point_t& b) {
        float dist = 0.0f;
        for (int i = 0; i < dim; i++) {
            float diff = a[i] - b[i];
            if (!is_single_bandwidth) { diff *= _get(bin_size_inv, i); }
            dist += diff * diff;
        }
        return dist <= (is_single_bandwidth ? _get(bandwidth, 0) * _get(bandwidth, 0) * convergence_tol_2 : convergence_tol_2);
    }
};

// We need to define the non-primitive-type static constexpr members outside the class
template<typename Params>
constexpr typename MeanShift<Params>::point_t MeanShift<Params>::min_bounds;
template<typename Params>
constexpr typename MeanShift<Params>::point_t MeanShift<Params>::max_bounds;
template<typename Params>
constexpr typename MeanShift<Params>::bandwidth_t MeanShift<Params>::exponent_factor;
template<typename Params>
constexpr typename MeanShift<Params>::bandwidth_t MeanShift<Params>::bin_size_inv;
template<typename Params>
constexpr std::array<int16_t, MeanShift<Params>::dim> MeanShift<Params>::hist_shape;
