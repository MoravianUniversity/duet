import os
import re
import sys
import random
import time
from multiprocessing import Pool

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.model_selection import ParameterGrid

from source_generation import compute_attenuation, combine_sources, load_wav_files
from duet_ms import DuetMS

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

SAMPLE_RATE = 16000    # sample rate of audio

NUM_SAMPLES = 10       # number of audio samples to generate
MIN_AUDIO_SUBSAMPLES = 2 # minimum number of audio subsamples to combine into a single sample
MAX_AUDIO_SUBSAMPLES = 2 # maximum number of audio subsamples to combine into a single sample
PAD_LENGTH = 800       # 50ms at 16kHz - avoid the beginning and end of the audio samples
SEGMENT_LENGTH = 4800  # 300ms at 16kHz (>144*2) - length of audio segments to use for evaluation
MIN_DELAY = -8         # minimum delay in samples
MAX_DELAY = 8          # maximum delay in samples
MIN_ATTEN = -1         # minimum symmetric attenuation
MAX_ATTEN = 1          # maximum symmetric attenuation

DATA: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

NUM_CORES = (os.process_cpu_count() or 2) // 2

OUTPUT_FILENAME = "duet_opt_results.csv"

grid = {
    # Audio Lengths to try (in ms)
    # Original target was 96ms, must be multiple of 16ms
    # In theory, longer audio lengths should give better results, but take longer to compute and use more memory
    "audio_length": [64, 80, 96],  # TODO: add 112

    # STFT window sizes to try
    # Must be power of 2, larger windows give better frequency resolution but worse time resolution
    "window": [128, 256],  # TODO: try [64, 128, 256, 512, 1024]

    # Oversampling factors to try
    # Must be odd integers with 1 being no oversampling
    # Higher oversampling improves resolution in STFT but is much slower
    "oversample": [1, 3],  # [1, 3, 5],  # higher is much, much, slower and not typically better

    # Thresholds to try for peak picking
    # Lower thresholds include more data but may include more noise and will be slower
    "threshold": [0.25, 0.35, 0.5, 0.75],  # [0.05, 0.1, 0.2, 0.25, 0.5, 0.75]

    # Bandwidths to try for mean-shift
    # Single values are isotropic, tuples are (time_bandwidth, frequency_bandwidth)
    # Bandwidths that are too small may lead to overfitting/noise, too large may miss sources
    # This is one of the most important parameters to tune and is likely anisotropic
    # This also effects grid size and thus memory usage and speed
    "bandwidth": [
        0.1, 0.2, 0.25, 0.3, 0.4,
        (0.1, 0.2), (0.1, 0.25), (0.1, 0.3), (0.1, 0.4),
        (0.2, 0.1), (0.2, 0.25), (0.2, 0.3), (0.2, 0.4),
        (0.25, 0.1), (0.25, 0.2), (0.25, 0.3), (0.25, 0.4),
        (0.3, 0.1), (0.3, 0.2), (0.3, 0.25), (0.3, 0.4),
        (0.4, 0.1), (0.4, 0.2), (0.4, 0.25), (0.4, 0.3),
    ],

    # Alpha max values to try, hardcoded for now
    "alpha_max": [1.5],  # TODO:
    # Delta max values to try, hardcoded for now
    "delta_max": [8.5],  # TODO:

    # Seed counts to try for mean-shift initialization
    # More seeds means that more unique peaks can possible be found (which may or may not be noise), but is slower
    # None means to use all possible unique peaks
    "seed_count": [20, 25, 35],            # [10, 25, 50, 75, 100] # None is slow and bad

    # Minimum bin counts to try for mean-shift seed selection
    # Larger will eliminate more seeds and be faster, but may miss some sources (but also may eliminate noise peaks)
    # A value of 1 means no elimination
    # Note that as bandwith increases, the size of bins increases and thus the bin count needs to increase to have the same effect
    "min_bin_count": [5, 10],              # [1, 5, 10, 25, 50, 100]  # in general, larger is much worse and only slightly faster (but there are others that are just as fast)

    # Max filter sizes to try for mean-shift seed selection
    # Must be None (for no filtering) or odd integers >1 for filtering
    # Remove possible seeds that are not local maxima within max_filter_size; this can help speed up results a lot by removing seeds
    # As this is increased, seed_count should be decreased or min_bin_count increased to prevent finding random local maxima that are not sources
    "max_filter_size": [None, 3],  # [None, 3, 5]

    # Convergence tolerances to try for mean-shift
    # Larger tolerances will converge faster but may be less accurate
    # A value of 1.0 means it will converge once reaching the nearest grid point
    "convergence_tol": [0.2, 0.25, 0.3],   # [0.05, 0.1, 0.2, 0.25, 0.5]

    # Alpha conversion operations to try
    # "symmetric" is the original DUET operation, "log" is a logarithmic scaling
    # "none" is not recommended as it makes no sense
    "alpha_op": ["symmetric", "log"],

    # Big delay methods to try
    # "none" is the original DUET method (will not work in our system)
    # "diff" is a differential method
    "big_delay": ["diff"],

    # Delta smoothing parameters to try
    # Smoothing helps reduce noise in delta estimates and is especially important for big-delta
    # Tuples are (freq, time) smoothing kernel sizes; (1, 1) means no smoothing
    "delta_smoothing": [(3, 1), (3, 3), (5, 1), (5, 3)],  # (1, 1), (3, 1), (3, 3), (5, 1), (5, 3)

    # Delta smoothing modes to try
    # Can be "mean", "median", or "gaussian"
    "delta_smoothing_mode": ["mean", "median", "gaussian"],

    # p and q weights for attenuation and delay estimators (from paper p.225)
    "p": [0.5, 1, 2],
    "q": [0, 2],
}

def find_alpha_deltas(x: np.ndarray, fs: int = 16000, **params):
    """
    Find alpha and delta peaks for given audio and DUET parameters.
    """
    # Extract audio length and trim audio
    audio_length = params.pop("audio_length")
    x = x[:, :audio_length*fs//1000]

    # Initialize DUET and compute peaks
    duet = DuetMS(fs, **params)
    xx = duet._normalize_data(x)
    _, tf_weights, alpha, delta = duet._compute_all(xx)
    alpha_peaks, delta_peaks = duet._find_peaks(tf_weights, alpha, delta)
    alpha_peaks = duet._convert_alpha_to_atn(alpha_peaks)

    # Return results
    return alpha_peaks, delta_peaks

def rmse(val):
    """
    Compute root mean square error of given values.
    """
    return np.sqrt(np.mean(np.multiply(val, val)))

def score_alpha_deltas(true, pred):
    """
    Score predicted alpha and delta values against true values.
    """
    true = np.transpose(true)
    pred = np.transpose(pred)
    if len(true) == 0 or len(pred) == 0:
        return {
            "score_custom": float('inf'),
            "score_custom2": float('inf'),
            "score_ari": 0.0,
            "score_nmi": 0.0,
            "score_vm": 0.0,
        }
    dists = sp.spatial.distance_matrix(true, pred)

    # TODO: try other scoring methods
    min1 = dists.min(1)
    min0 = dists.min(0)

    score_custom = (rmse(min1) + rmse(min0)) * (0.01 * np.exp(len(pred) - 5*len(true)) + 1)
    score_custom2 = rmse(min1) + rmse(min0) + max(min1.max(), min0.max())
    true_labels = np.arange(len(true))
    true_assigned = np.argmin(dists, axis=1)
    try:
        score_ari = adjusted_rand_score(true_labels, true_assigned)
        score_nmi = normalized_mutual_info_score(true_labels, true_assigned)
        score_vm = v_measure_score(true_labels, true_assigned)
    except Exception:
        score_ari = score_nmi = score_vm = 0.0
    return {
        "score_custom": score_custom,
        "score_custom2": score_custom2,
        "score_ari": score_ari,
        "score_nmi": score_nmi,
        "score_vm": score_vm,
    }
    # return (rmse(min1) + rmse(min0)) * (0.01 * np.exp(len(pred) - 5*len(true)) + 1)  # as we get 5 times more predictions than true sources, we start penalizing majorly
    # # return rmse(min1) + rmse(min0) + max(min1.max(), min0.max())

def init_data():
    """
    Initialize data. Loads audio files and generates samples.
    """
    base_dir = "FOAMS_processed_audio"

    #load segmentation_info and create a dictionary
    seg_info = pd.read_csv("segmentation_info.csv")
    seg_dict = dict(zip(seg_info['id'], seg_info['label']))
    all_sources = load_wav_files(base_dir)

    # Generate samples
    rand = random.Random(42)
    for _ in range(NUM_SAMPLES):
        num_sources = rand.randint(MIN_AUDIO_SUBSAMPLES, MAX_AUDIO_SUBSAMPLES)
        sources = rand.sample(all_sources, num_sources)
        selected_files = [s[0] for s in sources]
        sources = [s[1] for s in sources]

        # Create test delays/attenuations
        delays = [rand.uniform(MIN_DELAY, MAX_DELAY) for _ in sources]
        sym_attens = [rand.uniform(MIN_ATTEN, MAX_ATTEN) for _ in sources]
        attenuations = [compute_attenuation(a) for a in sym_attens]
        max_delay = int(np.ceil(max(abs(d) for d in delays)))

        # Take random segments
        starts = [rand.randint(PAD_LENGTH, len(s) - PAD_LENGTH - SEGMENT_LENGTH) for s in sources]
        sources = [s[start-max_delay:start+SEGMENT_LENGTH+max_delay] for s, start in zip(sources, starts)]

        # Information about the sources for debugging
        print([(seg_dict[int(re.findall(r'\d+', f)[0])], s, a, d)
               for f, s, a, d in zip(selected_files, starts, attenuations, delays)])

        # Generate stereo channels
        audio = combine_sources(sources, delays, attenuations)
        audio = audio[:, max_delay:-max_delay]

        DATA.append((audio, attenuations, delays))

def check_params(params: dict, data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = DATA) -> tuple[float, float]:
    """
    Check a given set of DUET parameters by scoring them on the entire test data.

    Returns the overall RMSE score and the average time taken per sample.
    """
    # overall scoring: RMSE of all individual scores
    # individual scores: addition of:
    #    take RMSE of L2 distances between every predicted value and the closest true value - makes sure that every prediction is close to a true source
    #    take RMSE of L2 distances between every true value and the closest predicted value - makes sure that every true source is close to a prediction
    # TODO: problems are though that:
    #    it favors many more predictions than true sources because they average out the bad ones
    #    it doesn't penalize far away predictions as much as it should if there are enough close ones
    #    even the best solutions can be missing some of the true sources
    # maybe use some other form of mean? double count max?
    start = time.time()
    try:
        preds = [find_alpha_deltas(d[0], **params) for d in data]
        elapsed = (time.time() - start) / len(data)
        all_scores = [score_alpha_deltas(d[1:], pred) for d, pred in zip(data, preds)]
        # cur_scores = [score_alpha_deltas(d[1:], pred) for d, pred in zip(data, preds)]
        # return rmse(cur_scores), elapsed
       
        # Average each metric across all samples (ignoring inf)
        score_keys = ["score_custom", "score_custom2", "score_ari", "score_nmi", "score_vm"]
        result = {}
        for k in score_keys:
            vals = [s[k] for s in all_scores if s[k] != float('inf')]
            result[k] = float(np.nanmean(vals)) if vals else float('inf')
        
        return result, elapsed
    except Exception as e:
        print(f"Error occurred while evaluating params {params}: {e}", file=sys.stderr)
        return {k: np.nan for k in ["score_custom", "score_custom2", "score_ari", "score_nmi", "score_vm"]}, np.nan
        # return np.nan, np.nan

def main():
    # Run all parameter combinations
    param_grid = ParameterGrid(grid)
    param_grid_df = pd.DataFrame(param_grid)
    all_scores = []
    times = []
    overall_start = time.time()
    with Pool(NUM_CORES, init_data) as pool:
        for i, (score, elapsed) in enumerate(pool.imap(check_params, param_grid, 100)):
            if i % 1000 == 0 and i > 0:
                # Print progress
                elapsed = time.time() - overall_start
                perc = i / len(param_grid)
                per = elapsed / i * 1000
                print(f"Evaluating {i}/{len(param_grid)} {perc:.1%}; {round(per)}ms per eval; est remaining {per*(len(param_grid)-i)/(60*1000):.1f} min; best so far: {min(s['score_custom'] for s in all_scores):.2f}")
                # Save results
                results = param_grid_df.iloc[:i].copy()
                for k in ["score_custom", "score_custom2", "score_ari", "score_nmi", "score_vm"]:
                    results[k] = [s[k] for s in all_scores]
                results["time"] = times
                results.to_csv(OUTPUT_FILENAME, index=False)
            all_scores.append(score)
            times.append(elapsed)

    # Final message
    elapsed = time.time() - overall_start
    per = elapsed / len(param_grid) * 1000
    print(f"Done; {round(per)}ms per eval; total time: {elapsed/60:.1f} min; best score: {min(all_scores):.2f}")

    # Save results
    results = param_grid_df.iloc[:i].copy()
    for k in ["score_custom", "score_custom2", "score_ari", "score_nmi", "score_vm"]:
        results[k] = [s[k] for s in all_scores]
    results["time"] = times
    results.to_csv(OUTPUT_FILENAME, index=False)

if __name__ == "__main__":
    main()
