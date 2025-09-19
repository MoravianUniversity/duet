#pragma once

#include <stdint.h>


//#define DUET_OFFLINE_PREPROCESS // include this when compiling for offline pre-processing
// When defined, many compile-time constants become runtime constants controlled by duet_init().
// To change them, duet_deinit() must be called before calling duet_init() again.
// A few compile-time parameters are still used:
// - DUET_SAMPLE_RATE (default 16000)
// - REC_SAMPLE_RATE (default 48000)
// - REC_CHANNELS (default 2, currently must be 2)
// - REC_BITS_PER_SAMPLE (default 16, currently must be 16)
// TODO: ATTENUATION_MAX, DELAY_MAX, bandwidths, and other mean-shift params


#ifdef DUET_OFFLINE_PREPROCESS
typedef int esp_err_t;
#else
#include <esp_err.h>
#endif


///////////////////////////////////////////////
////////// DUET Algorithm Parameters //////////
///////////////////////////////////////////////
// Frequency of the audio signal being processed
// This is the sample rate of the audio signal being processed, not the original audio signal
// Increasing this will:
//   - increase the time resolution
//   - decrease the frequency resolution (same number of frequency bins, more frequencies)
//   - increase the memory usage and processing time
// It must be an even divisor of the recording sample rate.
#ifndef DUET_SAMPLE_RATE
#define DUET_SAMPLE_RATE 16000 // 16 kHz
#endif

#ifndef DUET_OFFLINE_PREPROCESS
// TODO: maybe try 16 kHz sample rate (third of 48000 Hz) and a window size of 512 (radix-4 compatible)
// That gives:
//  - 256 frequency bins
//  - 7 time slices for 96 ms of audio (1536 samples) [may want to increase the total time to increase the number of time slices]
//  - 1792 time-frequency bins
//  - 12.5 KB of global memory  (6 KB for FFT tables, 6.5 KB for DUET globals)
//  - 124+ KB of stack memory
//       12 KB for 96 ms of audio (2 channels, 16 kHz, float32)
//          not actually needed to be saved, could minimize it 3 time slices (down to <4 KB)
//       TODO: haven't updated the rest of these
//       0.5 KB for FFT computation
//       32 KB for spectrogram
//       24 KB for alpha, delta, weights
//       26+ KB for mean shift prepare          (move to heap and reduce in the vast majority of cases)
//       10 KB for mean shift core              (plan to remove 8 KB of this)
//       20 KB for demixed (~10 samples)
//       at least 1KB for other things
//  Overall that is 27% of the SRAM!
//  Audio buffers take 18.4 KB (2 channels, 48 kHz, uint16, 96 ms of audio)
//  Still need to fit the AI model

// ESP32: 512 KB of SRAM (direct) at 960MB / second (~1000 bytes/us)
// ESP32-S3:
//   - 512 KB of SRAM (direct) at 960MB / second (~1000 bytes/us)
//   - 16 MB of PSRAM (indirect) at 40MB / second (~42 bytes/us) (half that for writes)
// ESP32-P4:
//   - 768 KB of SRAM (direct)
//   - 32 MB of PSRAM (indirect)

// Size of the STFT Window
// Must be a power of 2, max of 8192
// Preferred to be twice a power of 4 (8, 32, 128, 512, 2048, 8192) which will be relatively faster
// Increasing this will increase the frequency resolution but:
//   - decrease the time resolution
//   - increase the latency
//   - increase the memory usage and processing time
#ifndef DUET_WINDOW_SIZE
#define DUET_WINDOW_SIZE 256
#endif
#define DUET_WINDOW_SIZE_HALF (DUET_WINDOW_SIZE / 2) // Half the window size
#define DUET_N_FREQ (DUET_WINDOW_SIZE_HALF) // Number of frequency bins in the STFT (half of the window size)

// Number of audio samples we will save to do identification
// Must be a multiple of WINDOW_SIZE/2
// 100 ms * SAMPLE_RATE (16000) => 1600 samples => 1536 samples (96 ms) when rounded to the nearest multiple of WINDOW_SIZE/2
#ifndef DUET_N_SAMPLES
#define DUET_N_SAMPLES (DUET_SAMPLE_RATE / 10 / DUET_WINDOW_SIZE_HALF * DUET_WINDOW_SIZE_HALF) //  divide by 10 = 100 ms (10 Hz)
#endif
#define DUET_N_TIME (DUET_N_SAMPLES / DUET_WINDOW_SIZE_HALF + 1) // Number of time slices in the STFT

// The symmetric attenuation estimator value weights
// See the paper for more details. The value of 1 reduces the math needed to compute the weights.
#ifndef DUET_P
#define DUET_P 1.0f
#endif

// DUET algorithm: the delay estimator value weights
// See the paper for more details. The value of 0 reduces the math needed to compute the weights.
#ifndef DUET_Q
#define DUET_Q 0.0f
#endif

// Threshold to filter the points in the spectrogram. The higher this value,
// the faster it will run, but it may also start moving the cluster centers
// around. A recommendation is 0.05.
#ifndef DUET_POINT_THRESHOLD
#define DUET_POINT_THRESHOLD 0.5f
#endif

#endif

// Min and max bounds for processing attenuation (alpha) values
#ifndef ATTENUATION_MAX
#define ATTENUATION_MAX 3.6f
#endif

// Min and max bounds for processing delay (delta) values
#ifndef DELAY_MAX
#define DELAY_MAX 3.6f  // TODO: or 0.7f?
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initializes the Duet audio processing library.
 * This function must be called before using any other functions in the library.
 * It initializes the FFT library, precomputes numerous values, and sets up the
 * necessary parameters for the DUET algorithm along with allocating memory for
 * various buffers.
 *
 * When compiled with DUET_OFFLINE_PREPROCESS, this function takes many of the
 * DUET algorithm parameters as input instead of making them compile-time
 * constants.
 *
 * Returns ESP_OK on success, or an error code on failure.
 */
esp_err_t duet_init(
#ifdef DUET_OFFLINE_PREPROCESS
    int window_size, // size of the STFT window, must be a power of 2 between 8 and 8192 (inclusive); e.g. 256
    int n_samples,   // number of audio samples to use for identification, must be a multiple of window_size/2; e.g. 1536
    float p,         // symmetric attenuation estimator value weight; e.g. 1.0f
    float q,         // delay estimator value weight; e.g. 0.0f
    float point_threshold // point threshold for filtering; e.g. 0.5f
#endif
);

/**
 * Deinitialize the DUET audio processing library. This frees any resources
 * that were allocated by duet_init() and resets the state of the library.
 */
void duet_deinit();

/**
 * Reset the DUET state for completely new, unrelated, audio. Does not
 * deallocate or deinitialize any buffers or precomputed values.
 */
void duet_reset();

#ifdef DUET_OFFLINE_PREPROCESS
#ifdef __cplusplus
#include <complex>
typedef std::complex<float> cfloat;
#else
#include <complex.h>
typedef float _Complex cfloat;
#endif

/**
 * Add the new audio frame to the existing audio buffer and process it with
 * DUET. The new audio frame is interleaved channel data with
 * `AUDIO_FRAME_INIT_SIZE` samples for each channel.
 *
 * This returns a reference to the demixed sources of shape (n_sources, N_FREQ,
 * N_TIME) i.e. for each source there is a mono-spectrogram across all active
 * audio frames. The reference is valid until the next call to
 * `process_audio_frame()`.
 *
 * Due to incremental processing, the first N_TIME-1 calls to
 * `process_audio_frame()` will contain lots of zeros. This includes after init
 * and after a reset.
 */
const cfloat* duet_process_audio_frame(const float * const frame, int* n_sources);

int duet_get_n_channels();
int duet_get_n_freq();
int duet_get_n_time();
int duet_get_audio_frame_size();

#endif

#ifdef __cplusplus
}
#endif



// TODO: remove this and only support the overall function which calls these in the right order

// #include <complex.h>
// typedef float _Complex cfloat; // complex float type for the DUET algorithm

// void prep_data(const int16_t * const in, const int n, float* out);
// void decimate(const float* const input, int n, float* output, int offset);
// void decimate_alt(
//     const float* const input, // in, shape (N_CHANNELS, n)
//     int n,                    // number of samples in the input
//     float** output,           // out, shape (N_CHANNELS, offset + n/DECIMATION)
//     int offset                // offset in the output array to start writing at
// );
// void compute_spectrogram(
//     const float* const x, // in, shape of (N_CHANNELS, N_SAMPLES)
//     int new_times,
//     cfloat* out           // out, shape of (N_CHANNELS, N_FREQ, N_TIME)
// );
// void compute_spectrogram_alt(
//     float** const x,  // in, shape of (N_CHANNELS, N_SAMPLES)
//     int new_times,
//     cfloat* out             // out, shape of (N_CHANNELS, N_FREQ, N_TIME)
// );
// void compute_atten_and_delay(
//     const cfloat * const spectrogram, // in, shape (N_CHANNELS, N_FREQ, N_TIME)
//     const int new_times,
//     float* alpha,                     // out, shape (N_CHANNELS-1, N_FREQ, N_TIME)
//     float* delta                      // out, shape (N_CHANNELS-1, N_FREQ, N_TIME)
// );
// void compute_weights(
//     const cfloat * const spectrogram, // in, shape (N_CHANNELS, N_FREQ, N_TIME)
//     const int new_times,
//     float* tf_weights                 // out, shape (N_CHANNELS-1, N_FREQ, N_TIME)
// );
// #include <vector>
// void find_peaks(
//     const float * const tf_weights, // in, shape (N_CHANNELS-1, N_FREQ, N_TIME)
//     const float * const alpha,      // in, shape (N_CHANNELS-1, N_FREQ, N_TIME)
//     const float * const delta,      // in, shape (N_CHANNELS-1, N_FREQ, N_TIME)
//     std::vector<float>& alpha_peaks, // out, shape (n_sources, N_CHANNELS-1)
//     std::vector<float>& delta_peaks  // out, shape (n_sources, N_CHANNELS-1)
// );
// void convert_sym_to_atn(std::vector<float>& atn);
// void full_demix(
//     const cfloat * const spectrogram, // in, shape (2, N_FREQ, N_TIME)
//     const std::vector<float> &alpha,  // in, shape (n_sources, N_CHANNELS-1)
//     const std::vector<float> &delta,  // in, shape (n_sources, N_CHANNELS-1)
//     std::vector<cfloat> &demixed,     // out, shape (n_sources, N_FREQ, N_TIME)
//     uint8_t* best                     // out, shape (N_FREQ, N_TIME)
// );
