import os
import re
import sys
import random
import time
from collections import namedtuple
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
SEGMENT_LENGTH = 2400  # 150ms at 16kHz (>=144ms) - length of audio segments to use for evaluation
MIN_DELAY = -8         # minimum delay in samples
MAX_DELAY = 8          # maximum delay in samples
MIN_ATTEN = -1         # minimum symmetric attenuation
MAX_ATTEN = 1          # maximum symmetric attenuation
MIN_RMS_DB = -40       # minimum RMS in dB, used to filter out very quiet sources that will be indistinguishable from noise

# List of data samples, where each sample is a tuple of:
#   * full audio of all samples combined
#   * true attenuations (list/array of floats)
#   * true delays (list/array of floats)
#   * sub_audios (list of individual audio sources, composed with the same delays and attenuations as the full audio)
DATA: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]] = []

NUM_CORES = (os.process_cpu_count() or 2) // 2

OUTPUT_FILENAME = "duet_opt_results.csv"

ScoreParams = namedtuple("ScoreParams", ["true", "pred", "min0", "min1", "true_labels", "true_assigned"])

SCORERS = {
    "custom": lambda props: (rmse(props.min1) + rmse(props.min0)) * (0.01 * np.exp(len(props.pred) - 5*len(props.true)) + 1),
    "custom2": lambda props: rmse(props.min1) + rmse(props.min0) + max(props.min1.max(), props.min0.max()),
    "ari": lambda props: adjusted_rand_score(props.true_labels, props.true_assigned),
    "nmi": lambda props: normalized_mutual_info_score(props.true_labels, props.true_assigned),
    "vm": lambda props: v_measure_score(props.true_labels, props.true_assigned),
    "precision": lambda props: float(np.mean(props.min0 < 1.0)),
    "recall": lambda props: float(np.mean(props.min1 < 1.0)),
}

DEFAULT_SCORES = {
    "custom": float('inf'),
    "custom2": float('inf'),
    "ari": 0.0,
    "nmi": 0.0,
    "vm": 0.0,
    "precision": 0.0,
    "recall": 0.0,
}

grid = [{
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

    # Compute seeds with weights or not
    # Using weights may help find better seeds, but will require a dramatically different min_bin_count
    # (in my tests of a full audio sample, the `min_bin_count` changed from 50 to 600 when using weights to capture the same number of seeds)
    "compute_seeds_using_weights": [False, True],

    # Convergence tolerances to try for mean-shift
    # Larger tolerances will converge faster but may be less accurate
    # A value of 1.0 means it will converge once reaching the nearest grid point
    # (but since seeds are initialized at grid points, this means they will not move at all)
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
}]

def find_alpha_deltas(x: np.ndarray, fs: int = 16000, **params):
    """
    Find alpha and delta peaks for given audio and DUET parameters.
    """
    # Extract audio length and trim audio
    params = params.copy()
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

def score_alpha_deltas(true_data: tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]], pred):
    """
    Score predicted alpha and delta values against true values. The true values
    are a tuple of:
      * full audio of all samples combined
      * true attenuations (list/array of floats)
      * true delays (list/array of floats)
      * sub_audios (list of individual audio sources, composed with the same delays and attenuations as the full audio)
    """
    true_audio, true_attenuations, true_delays, sub_audios = true_data
    true = np.transpose([true_attenuations, true_delays])
    pred = np.transpose(pred)
    if len(true) == 0 or len(pred) == 0:
        return DEFAULT_SCORES.copy()
    dists = sp.spatial.distance.sdist(true, pred)

    # TODO: try other scoring methods
    min1 = dists.min(1)
    min0 = dists.min(0)
    true_labels = np.arange(len(true))
    true_assigned = np.argmin(dists, axis=1)
    params = ScoreParams(true, pred, min0, min1, true_labels, true_assigned)

    scores = {}
    for key, scorer in SCORERS.items():
        try:
            scores[key] = scorer(params)
        except Exception:
            scores[key] = DEFAULT_SCORES[key]
    return scores

def rms(x):
    """
    Compute root mean square of given values.
    """
    x = x - np.mean(x)  # remove DC offset
    return np.sqrt(np.mean(np.multiply(x, x)))

def rms_db(x):
    """
    Compute root mean square in dB of given values. If the original data is in the range [-1, 1],
    then this will be a non-positive value with more negative values indicating quieter sounds.
    """
    return 20 * np.log10(rms(x))

def int_delay(d: float) -> int:
    """
    Compute integer delay from a possibly fractional delay by rounding to the nearest integer.
    """
    return int(np.ceil(abs(d)))

def init_data():
    """
    Initialize data. Loads audio files and generates samples.
    """
    DATA.clear()  # clear data in case of re-initialization
    base_dir = "FOAMS_processed_audio"

    #load segmentation_info and create a dictionary
    seg_info = pd.read_csv("segmentation_info.csv")
    seg_dict = dict(zip(seg_info['id'], seg_info['label']))
    all_sources = load_wav_files(base_dir)

    # Generate samples
    rand = random.Random(42)
    while len(DATA) < NUM_SAMPLES:
        num_sources = rand.randint(MIN_AUDIO_SUBSAMPLES, MAX_AUDIO_SUBSAMPLES)
        sources = rand.sample(all_sources, num_sources)
        selected_files = [s[0] for s in sources]
        sources = [s[1] for s in sources]

        # Create test delays/attenuations
        delays = [rand.uniform(MIN_DELAY, MAX_DELAY) for _ in sources]
        sym_attens = [rand.uniform(MIN_ATTEN, MAX_ATTEN) for _ in sources]
        attenuations = [float(compute_attenuation(a)) for a in sym_attens]
        max_delay = max(int_delay(d) for d in delays)

        # Take random segments
        starts = [rand.randint(PAD_LENGTH, len(s) - PAD_LENGTH - SEGMENT_LENGTH) for s in sources]
        sources = [s[start-max_delay:start+SEGMENT_LENGTH+max_delay] for s, start in zip(sources, starts)]
        rms_values = [rms_db(s) for s in sources]
        if any(rms_db(s) < MIN_RMS_DB for s in sources):
            continue  # skip samples with very quiet sources that will be indistinguishable from noise

        # Information about the sources for debugging
        print([f"{seg_dict[int(re.findall(r'\d+', f)[0])]}: s={s}, a={a:.2f}, d={d:.2f}, rms={rms:.2f}"
               for f, s, a, d, rms in zip(selected_files, starts, attenuations, delays, rms_values)])

        # Generate stereo channels
        audio = combine_sources(sources, delays, attenuations)
        audio = audio[:, max_delay:-max_delay]

        # Generate individual parts for scoring
        # TODO: the trimming needs work...
        sub_audios = [combine_sources([s], [d], [a])[:, (max_delay-int_delay(d)):-int_delay(d)]
                      for s, d, a in zip(sources, delays, attenuations)]

        DATA.append((audio, attenuations, delays, sub_audios))

def check_params(params: dict, data: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]] = DATA) -> tuple[dict[str, float], float]:
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
        all_scores = [score_alpha_deltas(d, pred) for d, pred in zip(data, preds)]
       
        # Average each metric across all samples (ignoring inf)
        result = {}
        for k in SCORERS.keys():
            vals = [s[k] for s in all_scores if s[k] != float('inf') and not np.isnan(s[k])]
            if vals:
                result[f"{k}_mean"] = float(np.mean(vals))
                result[f"{k}_median"] = float(np.median(vals))
                result[f"{k}_min"] = float(np.min(vals))
                result[f"{k}_max"] = float(np.max(vals))
                result[f"{k}_25"] = float(np.percentile(vals, 25))
                result[f"{k}_75"] = float(np.percentile(vals, 75))
            else:
                result[f"{k}_mean"] = float('inf')
                result[f"{k}_median"] = float('inf')
                result[f"{k}_min"] = float('inf')
                result[f"{k}_max"] = float('inf')
                result[f"{k}_25"] = float('inf')
                result[f"{k}_75"] = float('inf')



            # TODO: use rmse(cur_scores) instead of nanmean?
        
        return result, elapsed
    except Exception as e:
        print(f"Error occurred while evaluating params {params}: {e}", file=sys.stderr)
        return {k: np.nan for k in SCORERS.keys()}, np.nan

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
                print(f"Evaluating {i}/{len(param_grid)} {perc:.1%}; "
                      f"{round(per)}ms per eval; "
                      f"est remaining {per*(len(param_grid)-i)/(60*1000):.1f} min; "
                      f"best so far: {min(s['custom'] for s in all_scores):.2f}")
                # Save results
                results = param_grid_df.iloc[:i].copy()
                for k in SCORERS.keys():
                    results[f"score_{k}"] = [s[k] for s in all_scores]
                results["time"] = times
                results.to_csv(OUTPUT_FILENAME, index=False)
            all_scores.append(score)
            times.append(elapsed)

    # Final message
    elapsed = time.time() - overall_start
    per = elapsed / len(param_grid) * 1000
    print(f"Done; {round(per)}ms per eval; total time: "
          f"{elapsed/60:.1f} min; "
          f"best score: {min(all_scores):.2f}")

    # Save results
    results = param_grid_df.iloc[:i].copy()
    for k in SCORERS.keys():
        results[f"score_{k}"] = [s[k] for s in all_scores]
    results["time"] = times
    results.to_csv(OUTPUT_FILENAME, index=False)

if __name__ == "__main__":
    main()
