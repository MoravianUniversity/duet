import os
import csv
import random

import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

def compute_attenuation(symmetric_value: float) -> float:
    """
    Compute attenuation based on a symmetric attenuation value.

    Parameters
    ----------
    symmetric_value : float
        Input symmetric attenuation.

    Returns
    -------
    float
        Computed attenuation coefficient.
    """
    return (symmetric_value + np.sqrt(symmetric_value * symmetric_value + 4)) / 2


def combine_sources(sources: list[np.ndarray], delays: list[float], attenuations: list[float], delay_res: int = 20) -> np.ndarray:
    """
    Combine multiple audio sources into a single two-channel signal by summing them.

    Parameters
    ----------
    sources : list[np.ndarray]
        List of audio sources (normalized arrays).
    delays : list[float]
        Delay for each source in samples (can be fractional, accurate up to 1/delay_res).
    attenuations : list[float]
        Attenuation factor for each source.
    delay_res : int, optional
        Resolution for fractional delays. Default is 20. Higher values allow finer delay
        adjustments, but greatly increase computation time.

    Returns
    -------
    np.ndarray
        Combined signal for both channels.
    """
    channel_1 = np.zeros_like(sources[0])
    channel_2 = np.zeros_like(sources[0])
    max_len = 0
    for src, delay, atten in zip(sources, delays, attenuations):
        src = signal.resample(src, len(src) * delay_res)  # Upsample to allow for fractional delays
        delay = round(delay * delay_res)  # Convert delay to integer samples in upsampled space
        if delay > 0:
            ch1, ch2 = src[delay:], src[:-delay]
        elif delay < 0:
            ch1, ch2 = src[:delay], src[-delay:]
        else:
            ch1, ch2 = src, src
        ch1 = signal.resample(ch1, len(ch1) // delay_res)  # Downsample back to original rate
        ch2 = signal.resample(ch2, len(ch2) // delay_res)
        channel_1[:len(ch1)] += ch1
        channel_2[:len(ch2)] += ch2 * atten
        max_len = max(max_len, len(ch1), len(ch2))
    channel_1 = channel_1[:max_len]
    channel_2 = channel_2[:max_len]
    return normalize_audio(np.vstack((channel_1, channel_2)))


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to the range [-1, 1].

    Parameters
    ----------
    audio : np.ndarray
        Input audio array.

    Returns
    -------
    np.ndarray
        Normalized audio array.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def load_wav_files(base_dir: str, frequency: int = 16000) -> list[tuple[str, np.ndarray]]:
    """
    Load and normalize all .wav files to be the same frequency and floating point from -1 to 1.

    Parameters
    ----------
    base_dir : str
        Directory containing WAV files (possibly with subdirectories).
    frequency : int, optional
        Target frequency for all of the files

    Returns
    -------
    list[tuple[str, np.ndarray]]
        List of tuples containing file paths and normalized source arrays.
    """
    wav_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(base_dir)
        for file in files if file.endswith(".wav")
    ]

    # Load and normalize
    sources = []
    for f in wav_files:
        fs, data = wav.read(f)
        if fs != frequency:
            data = signal.resample(data, frequency * data.shape[0] // fs)
        if data.dtype.kind != 'f':
            data = data / np.iinfo(data.dtype).max
        else:
            # TODO: in my tests, this is likely 32768 since it seems to be 16-bit audio converted to float
            data = data / np.max(np.abs(data))  # Normalize to -1 to 1 if already float
        if data.ndim > 1:
            # Handle multi-channel audio by taking each channel as a separate source
            for ch in range(data.shape[1]):
                sources.append((f"{f}#ch{ch}", data[:, ch]))
        else:
            sources.append((f, data))

    return sources


def save_wav(filename: str, audio: np.ndarray, frequency: int = 16000) -> None:
    """
    Save two mono channels as a stereo WAV file.

    Parameters
    ----------
    filename : str
        Output filename.
    audio : np.ndarray
        2D array with shape (2, N) where the first row is channel 1 and the second row is channel 2.
    frequency : int, optional
        Sampling frequency.
    """
    stereo = (normalize_audio(audio.T) * 32767).astype(np.int16)
    wav.write(filename, frequency, stereo)


def plot_sources_and_output(j, sources: list[np.ndarray], output_path: str = "output.wav") -> None:
    """
    Plot individual sources and stereo output waveform.

    Parameters
    ----------
    sources : list[np.ndarray]
        List of audio sources.
    output_path : str, optional
        Path to the stereo output file.
    """
    sr, stereo_data = wav.read(output_path)
    stereo_data = normalize_audio(stereo_data.astype(float))
    total_plots = len(sources) + 1
    plt.figure(figsize=(14, 10))

    # Plot each source
    for i, src in enumerate(sources):
        plt.subplot(total_plots, 1, i + 1)
        plt.plot(src, linewidth=0.7)
        plt.title(f"Source {i + 1}")
        plt.ylabel("Amplitude")
        plt.xlabel("Samples")

    # Plot output channels
    plt.subplot(total_plots, 1, total_plots)
    plt.plot(stereo_data[:, 0], label="Channel 1", alpha=0.7)
    plt.plot(stereo_data[:, 1], label="Channel 2", alpha=0.7)
    plt.title("Output Stereo Channels")
    plt.ylabel("Amplitude")
    plt.xlabel("Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/plot_{j + 1}.png")
    #plt.show()


from duet_ms import DuetMS

def find_alpha_deltas(x: np.ndarray,
                      real_alpha: list[float], real_delta: list[float],
                      fs: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the alpha and delta values using DUET.

    Parameters
    ----------
    x : np.ndarray
        2D array with shape (2, N) where the first row is channel 1 and the second row is channel 2.
    fs : int, optional
        Sampling frequency, default is 16000.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing alpha peaks and delta peaks.
    """
    duet = DuetMS(fs, threshold=0.5, alpha_max=1.25, delta_max=8.25, bandwidth=[0.05, 0.5])
    xx = duet._normalize_data(x)
    tf, tf_weights, alpha, delta = duet._compute_all(xx)
    cutoff = None  # 63 is halfway point
    alpha_peaks, delta_peaks = duet._find_peaks(tf_weights[:cutoff], alpha[:cutoff], delta[:cutoff])

    # Transform the weights for display purposes
    weights_log = np.log(tf_weights)
    weights_log = (weights_log - np.min(weights_log)) / (np.max(weights_log) - np.min(weights_log)) / 5

    #print(alpha.shape, delta.shape, alpha_peaks.shape, delta_peaks.shape)

    print(f"True Sym Attens: {real_alpha}")
    print(f"Est Sym Attens:  {alpha_peaks}")
    print(f"True Delays:     {real_delta}")
    print(f"Est Delays:      {delta_peaks}")

    tf_abs = np.abs(tf)
    mn, mx = np.min(tf_abs), np.max(tf_abs)

    plt.figure(figsize=(10, 10))
    plt.scatter(alpha[:cutoff], delta[:cutoff], s=1, alpha=weights_log[:cutoff])
    plt.scatter(sym_atn_peaks, delay_peaks, c='black', marker='x')
    for a, d in zip(real_alpha, real_delta):
        plt.scatter(a, d, marker='o', edgecolors='red', facecolors='none', s=80)
    plt.xlabel('Attenuation (alpha)')
    plt.ylabel('Delay (delta)')
    plt.xlim(-duet.alpha_max, duet.alpha_max)
    plt.ylim(-duet.delta_max, duet.delta_max)
    plt.show()

    return alpha_peaks, delta_peaks


def display_results(alpha: np.ndarray, delta: np.ndarray, tf_weights: np.ndarray,
                    pred_alpha: list[float], pred_delta: list[float],
                    real_alpha: list[float], real_delta: list[float]):
    # Transform the weights for display purposes
    weights_log = np.log(tf_weights)
    weights_log = (weights_log - np.min(weights_log)) / (np.max(weights_log) - np.min(weights_log)) / 5

    #plt.figure(figsize=(10, 10))
    plt.scatter(alpha, delta, s=1, alpha=weights_log)
    plt.scatter(pred_alpha, pred_delta, c='black', marker='x')
    for a, d in zip(real_alpha, real_delta):
        plt.scatter(a, d, marker='o', edgecolors='red', facecolors='none', s=80)
    plt.xlabel('Attenuation (alpha)')
    plt.ylabel('Delay (delta)')
    plt.xlim(-2, 2)
    plt.ylim(-8, 8)
    plt.show()


def main():
    random.seed(42)
    base_dir = "FOAMS_processed_audio"

    #load segmentation_info and create a dictionary 
    seg_info = pd.read_csv("segmentation_info.csv")
    seg_dict = dict(zip(seg_info['id'], seg_info['label']))
    all_sources = load_wav_files(base_dir)

    with open("alphadeltas.csv", "w") as f:
        writer = csv.writer(f)
        for i in range(10):
            print(f"--- Iteration {i + 1} ---")
            sources = random.sample(all_sources, 2)
            selected_files = [s[0] for s in sources]
            sources = [s[1] for s in sources]
            
            # Create test delays/attenuations
            # A 17 cm difference between microphones is about 0.5 ms apart (ears are typically about 15-17 cm apart)
            # At 16 kHz, that's about 8 samples apart, so we should limit delays to +/- 8 samples
            delays = [random.uniform(-8, 8) for _ in sources]  # TODO

            sym_attens = [random.uniform(-1, 1) for _ in sources]
            # sym attens of 0 means no attenuation (1.0) - both channels equal
            # sym attens from -0.7 to 0.7 map to attens from 0.7 to 1.4
            # sym attens from -1 to 1 map to attens from 0.6 to 1.6
            # sym attens from -2 to 2 map to attens from 0.4 to 2.4
            # sym attens from -3 to 3 map to attens from 0.3 to 3.3
            # sym attens from -3.6 to 3.6 map to attens from 0.26 to 3.8  (1/4 to 4x as amplitude between channels)
            attenuations = [compute_attenuation(a) for a in sym_attens]

            # Generate stereo channels
            audio = combine_sources(sources, delays, attenuations)

            # Save and visualize
            save_wav(f"outputs/output_{i + 1}.wav", audio)
            #plot_sources_and_output(i, sources, f"outputs/output_{i + 1}.wav")
          
            #get the label for each file
            selected_files_id = [int(os.path.basename(f).split('_')[0]) for f in selected_files]
            selected_files_labels = [seg_dict.get(id, "unknown") for id in selected_files_id]
            row = []
            for a, d , s, l in zip(attenuations, delays, selected_files, selected_files_labels):
                row += [i, a, d, s, l]
            writer.writerow(row)

            sym_atn_peaks, delay_peaks = find_alpha_deltas(audio, sym_attens, delays)


if __name__ == "__main__":
    main()
