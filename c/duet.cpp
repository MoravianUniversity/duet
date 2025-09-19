#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include <malloc.h>  // for memalign (on macOS this requires a special malloc.h)

#include <vector>
#include <algorithm>

#include <esp_dsp.h>

#include "duet.h"
#include "mean_shift.hpp"
#include "fast_math.hpp"



// Make sure the audio codec is using the settings we are expecting/support
#ifdef DUET_OFFLINE_PREPROCESS
#ifndef REC_SAMPLE_RATE
#define REC_SAMPLE_RATE 48000
#endif
#ifndef REC_CHANNELS
#define REC_CHANNELS 2
#endif
#ifndef REC_BITS_PER_SAMPLE
#define REC_BITS_PER_SAMPLE 16
#endif
#else
#include "audio.h"
#endif

static_assert(REC_BITS_PER_SAMPLE == 16, "DUET: BITS_PER_SAMPLE must be 16");
static_assert(REC_CHANNELS == 2, "DUET: CHANNELS must be 2 (for now)");
static_assert(REC_SAMPLE_RATE % DUET_SAMPLE_RATE == 0, "DUET: REC_SAMPLE_RATE must be a multiple of DUET_SAMPLE_RATE");

constexpr int N_CHANNELS = REC_CHANNELS;
constexpr int SAMPLE_RATE = DUET_SAMPLE_RATE; // 16 kHz


///////////////////////////////////////////
////////// Constants and Globals //////////
///////////////////////////////////////////

#ifdef DUET_OFFLINE_PREPROCESS
// Global runtime parameters instead of compile-time constants; see below for more details
static int WINDOW_SIZE;
static int N_SAMPLES;
static float P;
static float Q;
static float POINT_THRESHOLD;
static int WINDOW_SIZE_HALF;
static int HOP;
static bool IS_POWER_OF_FOUR;
static bool USE_RADIX_4;
static float* WINDOW = nullptr;      // 16-bit aligned, WINDOW_SIZE
static float* DUAL_WINDOW = nullptr; // 16-bit aligned, WINDOW_SIZE
static int N_FREQ;
static float* FREQUENCIES = nullptr; // 16-bit aligned, N_FREQ
static int N_TIME;
static int N_FREQ_TIME;
static float* FREQS_INV = nullptr;   // 16-bit aligned, N_FREQ
static float* FREQS_POW_Q = nullptr; // 16-bit aligned, N_FREQ
static int AUDIO_FRAME_INIT_SIZE;
#else

// Turn header defines into constants
constexpr int WINDOW_SIZE = DUET_WINDOW_SIZE;
constexpr int N_SAMPLES = DUET_N_SAMPLES;
constexpr float P = DUET_P;
constexpr float Q = DUET_Q;
constexpr float POINT_THRESHOLD = DUET_POINT_THRESHOLD;

// Check parameters
static_assert((WINDOW_SIZE != 0) && ((WINDOW_SIZE & (WINDOW_SIZE - 1)) == 0), "Window size must be a power of 2");
static_assert(WINDOW_SIZE >= 8 && WINDOW_SIZE <= 8192, "Window size must be at least 8 and at most 8192");
static_assert((N_SAMPLES != 0) && (((N_SAMPLES % (WINDOW_SIZE / 2))) == 0), "Number of samples must be a multiple of WINDOW_SIZE/2");

// Compute derived constants
constexpr int WINDOW_SIZE_HALF = WINDOW_SIZE >> 1;
constexpr int HOP = WINDOW_SIZE_HALF;
constexpr bool IS_POWER_OF_FOUR = (WINDOW_SIZE & 0xAAAAAAAA) == 0; // true if WINDOW_SIZE is a power of 4, false otherwise
constexpr bool USE_RADIX_4 = !IS_POWER_OF_FOUR; // use radix-4 FFT if the window size is twice a power of 4, otherwise use radix-2 FFT

// Hamming window coefficients
static __attribute__((aligned(16))) float WINDOW[WINDOW_SIZE];      // with WS = 256, this is 1 KB of memory

// The dual window coefficients for ISTFT
static __attribute__((aligned(16))) float DUAL_WINDOW[WINDOW_SIZE]; // with WS = 256, this is 1 KB of memory

// Number of frequency bins in the STFT (half of the window size)
constexpr int N_FREQ = WINDOW_SIZE_HALF;

// Frequencies in radians per second, starting from 2π/N_FREQ to 2π.    
static __attribute__((aligned(16))) float FREQUENCIES[N_FREQ];      // with WS = 256, this is 0.5 KB of memory

// Number of time slices in the STFT
constexpr int N_TIME = N_SAMPLES / WINDOW_SIZE_HALF + 1; // with WS = 256, this is 13 time slices

// Total number of time-frequency bins in the STFT
constexpr int N_FREQ_TIME = (N_FREQ * N_TIME); // with WS = 256 and 1536 samples, this is 1536 time-frequency bins

// Precomputed values for the DUET algorithm
static __attribute__((aligned(16))) float FREQS_INV[N_FREQ];   // 1 / FREQUENCIES           // with WS = 256, this is 0.5 KB of memory
static __attribute__((aligned(16))) float FREQS_POW_Q[(Q != 0.0f) ? N_FREQ : 0]; // pow(fabs(FREQUENCIES), Q) // with WS = 256, this is 0.5 KB of memory

#endif

// Decimation Parameters
// The decimation filter inherently has a delay of (DECIMATION_FIR_LEN-1)/2 samples in the original
// source (approximately (DECIMATION_FIR_LEN-1)/(2*DECIMATION) samples in the decimated output
// [however there can be a few extra delay samples due to the way the decimation is done]).
// We need to account for that delay when accounting for the latency.
// Example:
//   48000 kHz -> 16000 kHz decimation with a 60-tap FIR filter:
//   (60-1)/(2*48000) = 0.615 ms delay
// Thus, increasing the FIR filter length will increase the latency. It also increases 
// processing time and memory usage, so it is best to keep it as small as possible. However,
// smaller values result in more aliasing possibly making the audio more artificial sounding.
// A multiple of 4 has some slight benefits in processing time and memory usage. In scipy, the
// default is 20*DECIMATION + 1.
constexpr int DECIMATION = REC_SAMPLE_RATE / DUET_SAMPLE_RATE;
constexpr int DECIMATION_FIR_LEN = DECIMATION*20; // Length of the FIR filter for decimation (prefer a multiple of 4, although odd lengths have benefits as well)
static __attribute__((aligned(16))) float DECIMATION_FIR_COEFFS[(DECIMATION_FIR_LEN + 3) & ~3]; // with DECIMATION_FIR_LEN = 40, this is 0.16 KB of memory
static __attribute__((aligned(16))) fir_f32_t decimation_filters[N_CHANNELS];

#ifndef DUET_OFFLINE_PREPROCESS
// Number of samples given during each frame (pre-decimation)
constexpr int AUDIO_FRAME_INIT_SIZE = WINDOW_SIZE_HALF * DECIMATION;
#endif

// Mean Shift Parameters
// TODO: make most of these configurable with defines or at runtime
struct DuetMeanShiftParams: public MeanShiftParams {
    static constexpr int dim = (N_CHANNELS-1)*2;
    static constexpr float bandwidth = 0.2f;
    //static constexpr std::array<float, 2> bandwidth = { 0.2f, 0.2f };
    static constexpr std::array<float, 2> min_bounds = {-ATTENUATION_MAX, -DELAY_MAX};
    static constexpr std::array<float, 2> max_bounds = { ATTENUATION_MAX,  DELAY_MAX};
    static constexpr float convergence_tol = 0.1f;
    static constexpr int min_count = 3;
    static constexpr int top_n = 20;
};
// Mean Shift object for DUET and temporary vectors
typedef MeanShift<DuetMeanShiftParams> DuetMeanShift;
static DuetMeanShift mean_shift;
static std::vector<DuetMeanShift::point_t> ms_points;
static std::vector<float> ms_weights;
static std::vector<DuetMeanShift::point_t> ms_centroids;


/////////////////////////////
///////// Utilities /////////
/////////////////////////////
#define CHECK_ESP_DSP(x) do { esp_err_t _err = (x); if (unlikely(_err != ESP_OK)) return _err; } while (0)


/**
 * Compute the Hamming window coefficients for the given length.
 * The coefficients are stored in the `window` array.
 */
void dsps_wind_hamming_f32(float* window, int len) {
    float factor = DIVIDE(PI_2, (float)(len - 1));
    for (int i = 0; i < len; i++) { window[i] = 0.54f - 0.46f * cosf(i * factor); }
}

/**
 * Compute the FIR filter coefficients for an anti-aliasing filter for use with
 * decimation. This uses a Hamming window combined with a sinc function:
 * 
 *      hamming(i) * c * sinc(cutoff * i)
 * 
 * where `c` is the cutoff frequency, sinc(x) = sin(pi*x)/(pi*x), and `i` is the
 * index from -n/2 to n/2. After computing the coefficients, the values are
 * normalized so that the filter has a total gain of 1.
 */
void compute_fir_coeffs(
    float* coeffs,  // output, length n
    int n,          // length of the FIR filter
    float cutoff    // cutoff frequency (normalized, 0 to 1)
) {
    const float cutoff_pi = PI_ * cutoff, halfway = (n - 1) * 0.5f;
    float window[n];
    dsps_wind_hamming_f32(window, n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float c = 1.0f;
        if (i != halfway) {
            c = cutoff_pi * (i - halfway);
            c = DIVIDE(sinf(c), c); // sinc(x)
        }
        c *= window[i] * cutoff;
        coeffs[i] = c;
        sum += c;
    }
    float sum_inv = recip(sum);
    for (int i = 0; i < n; i++) { coeffs[i] *= sum_inv; }
}

/**
 * Roll the last `new_len` bytes of each `data` array of shape. The new and
 * overwrite data is uninitialized.
 * 
 * Example: to make room for a new float while also overwriting the last float
 * in an array of shape (128, 12):
 *    roll(data, 128, 12, 1, 1);
 */
template <typename T> void roll(
    T* data,            // in/out, shape of (shape, data_len)
    int shape,          // product of all dimensions except the last
    int data_len,       // number of elements in the last dimension
    int new_len,        // number of elements to have room for in the last dimension (not inc overwritten data)
    int overwrite_len   // number of elements to overwrite in the last dimension
) {
    for (int i = 0; i < shape; i++) {
        T* a = &data[i*data_len];
        memmove(a, &a[new_len], (data_len-new_len-overwrite_len)*sizeof(T));  // TODO: use dsps_memcpy?
    }
}

/**
 * Roll the last `new_len` bytes of each `data` array of shape. This assumes
 * that the new length is quite small compared to the total size of the data.
 * This does a memory copy of the entire buffer at once. The new data is
 * uninitialized.
 * 
 * Example: to make room for a new float while also overwriting the last float
 * in an array of shape (128, 12):
 *    roll2(data, 128*12, 1);
 */
template <typename T> void roll2(
    T* data,            // in/out, shape of (shape, data_len)
    int total_size,     // total size of the data
    int new_len         // number of elements to have room for in the last dimension (not inc overwritten data)
) {
    memmove(data, data+new_len, (total_size-new_len)*sizeof(T));  // TODO: use dsps_memcpy?
}

/**
 * Roll the last `new_len` elements of each `data` array of shape.
 * This assumes the data is part of a larger buffer and simply shifts the
 * pointer to the new position unless the buffer is full, in which case it
 * rolls the data back to the initial position. This results in much, much,
 * fewer memory copies and the few times it does copy, it copies the entire
 * buffer at once. The new data is uninitialized.
 * 
 * Example: to make room for a new float in an array of shape (128, 12) that
 * has a double-sized buffer:
 *    data = roll_with_buffer(data, buf, buf+2*1536, 1536, 1);
 */
template <typename T> T* roll_with_buffer(
    T* data,      // in/out, shape of (shape, data_len)
    T* initial,   // in/out, shape of (shape, data_len)
    T* end,       // in/out, shape of (shape, data_len)
    int total_size,     // total size of the data
    int new_len         // number of elements to have room for in the last dimension (not inc overwritten data)
) {
    if (data + total_size + new_len > end) {
        memmove(initial, data+new_len, (total_size-new_len)*sizeof(T));  // TODO: use dsps_memcpy?
        return initial;
    } else {
        data += new_len;
        return data;
    }
}

/**
 * Convert interleaved 16-bit signed integer audio data to a normalized float array.
 * The normalization is done by dividing each element by 32767.0f.
 * The input data is expected to be in the format: [left, right, left, right, ...].
 * The output data is in the format: [left, left, ..., right, right, ...].
 * The number of samples in a single channel is given as `n`.
 */
void OPTIMIZE_FOR_SPEED prep_data(const int16_t * const in, const int n, float* out) {
    float scale = 1.0f / 32767.0f;  // TODO: or 32768?
    for (int c = 0; c < N_CHANNELS; c++) {
        int16_t* in_chan = (int16_t*)&in[c];
        float* out_chan = &out[c * n];
        for (int i = 0; i < n; i++) {
            out_chan[i] = in_chan[i*N_CHANNELS] * scale;
        }
    }
}


////////////////////////////
///////// Decimate /////////
////////////////////////////

/**
 * Compute the global FIR coefficients for the decimation filter.
 */
void init_decimate_fir_coeffs() {
    constexpr int fir_len_4 = (DECIMATION_FIR_LEN + 3) & ~3;  // round up to the nearest multiple of 4 and zero-pad
    compute_fir_coeffs(DECIMATION_FIR_COEFFS, DECIMATION_FIR_LEN, recip(DECIMATION));
    for (int i = DECIMATION_FIR_LEN; i < fir_len_4; i++) { DECIMATION_FIR_COEFFS[i] = 0.0f; }
}

/**
 * Initialize the decimation filter for the given decimation factor and FIR
 * filter length. There must be a separate instance of the filter for each
 * channel, however, they can resuse the same coefficients so they save
 * some memory. You must call init_decimate_fir_coeffs() before calling this
 * function.
 * 
 * Also looked into using the signed-16-bit-integer FIR decimation functions
 * from the ESP-DSP library, but they are not as efficient as the float ones,
 * but they are surprisingly about half to a third of the speed.
 */
int init_decimate_filter(fir_f32_t* fir) {
    // dsps_fird_init_f32 on ESP32-S3 requires:
    //    a multiple of 4 fir length (pad with zeros if necessary)
    //    coeffs and delay line must be aligned to 16 bytes: memalign(16, nbytes) or __attribute__((aligned(16))) for global variables
    constexpr int fir_len_4 = (DECIMATION_FIR_LEN + 3) & ~3;  // round up to the nearest multiple of 4 and zero-pad
    float* delay = (float*)memalign(16, fir_len_4 * sizeof(float));
    if (delay == NULL) { return ESP_ERR_NO_MEM; }
    return dsps_fird_init_f32(fir, DECIMATION_FIR_COEFFS, delay, fir_len_4, DECIMATION);
}

/**
 * Deinitialize the decimation filter. This frees the delay line buffer and
 * sets the coefficients to NULL.
 */
void deinit_decimate_filter(fir_f32_t* fir) {
    if (fir) { free(fir->delay); fir->delay = NULL; fir->coeffs = NULL; }
}

/**
 * Decimate the input signal. This uses the decimation filter initialized with
 * init_decimate_filter() and the coefficients initialized with
 * init_decimate_fir_coeffs(). The result is written to the output array, with
 * each channel being offset.
 */
void decimate(
    const float* const input, // in, shape (N_CHANNELS, n)
    int n,                    // number of samples in the input
    float* output,            // out, shape (N_CHANNELS, offset + n/DECIMATION)
    int offset                // offset in the output array to start writing at
) {
    int output_n = n / DECIMATION;
    int output_n_total = output_n + offset;
    output = &output[offset]; // adjust output pointer to the offset
    for (int i = 0; i < N_CHANNELS; i++) {
        int output_count = dsps_fird_f32(&decimation_filters[i], &input[i*n], &output[i*output_n_total], output_n);
        // TODO: check output_count == output_n?
    }
}


void decimate_alt(
    const float* const input, // in, shape (N_CHANNELS, n)
    int n,                    // number of samples in the input
    float** output,           // out, shape (N_CHANNELS, offset + n/DECIMATION)
    int offset                // offset in the output array to start writing at
) {
    int output_n = n / DECIMATION;
    for (int i = 0; i < N_CHANNELS; i++) {
        float* out = output[i] + offset;
        int output_count = dsps_fird_f32(&decimation_filters[i], &input[i*n], out, output_n);
        // TODO: check output_count == output_n?
    }
}



///////////////////////////////////////
///////// Compute Spectrogram /////////
///////////////////////////////////////

/** Initialize the global STFT window, a Hamming window. */
void init_stft_window() { dsps_wind_hamming_f32(WINDOW, WINDOW_SIZE); }

/** Initialize the global STFT dual window. */
void init_stft_dual_window() {
    // assumes the window is symmetrical
    DUAL_WINDOW[HOP] = DIVIDE(WINDOW[HOP], WINDOW[HOP] * WINDOW[HOP] + WINDOW[0] * WINDOW[0]);
    for (int i = 0; i < HOP; i++) { DUAL_WINDOW[i] = DIVIDE(WINDOW[i], WINDOW[i] * WINDOW[i] + WINDOW[i+HOP] * WINDOW[i+HOP]); }
    for (int i = HOP; i < WINDOW_SIZE; i++) { DUAL_WINDOW[i] = DUAL_WINDOW[WINDOW_SIZE-i-1]; }
}

/**
 * Initialize the global frequency bins for the STFT.
 * The frequencies are in radians per second, starting from 2π/N_FREQ to 2π.
 */
void init_stft_freqs() {
    float factor = PI_ / N_FREQ;
    for (int i = 0; i < N_FREQ; i++) { FREQUENCIES[i] = (i + 1) * factor; }
}

/**
 * Initialize the global FFT library for the STFT.
 * This initializes the FFT libraries for both radix-2 and radix-4 FFTs.
 * Returns true if successful, false otherwise.
 */
esp_err_t init_stft_fft() {
    // the parameter is the number of complex numbers in the FFT
    // need to always initialize the radix-4 FFT since it is used for the bit-rev step
    CHECK_ESP_DSP(dsps_fft4r_init_fc32(NULL, N_FREQ));
    CHECK_ESP_DSP(dsps_fft2r_init_fc32(NULL, N_FREQ));
    return ESP_OK;
}

/**
 * Deinitialize the global FFT library for the STFT.
 * This deinitializes the FFT libraries for both radix-2 and radix-4 FFTs.
 */
void deinit_stft_fft() {
    dsps_fft2r_deinit_fc32();
    dsps_fft4r_deinit_fc32();
}

/**
 * Take the complex conjugate of the input array. Operated in-place.
 * There are n complex numbers in the array, each a pair of (real, imag) floats.
 */
void OPTIMIZE_FOR_SPEED array_conj(float* x, int n) {
    x += 1; // move to the imaginary part
    for (int i = 0; i < n; i++) { x[2*i] = -x[2*i]; }  // TODO: maybe use dsps_mulc_f32(x, x, n, -1f, 2, 2)
}

/**
 * Perform the FFT on the input signal. The signal is overwritten and either
 * has a length of N_FREQ complex numbers or 2*N_FREQ reals.
 */
esp_err_t OPTIMIZE_FOR_SPEED fft_core(float* x) {
    if (USE_RADIX_4) {
        CHECK_ESP_DSP(dsps_fft4r_fc32(x, N_FREQ));
        CHECK_ESP_DSP(dsps_bit_rev4r_fc32(x, N_FREQ));
    } else {
        CHECK_ESP_DSP(dsps_fft2r_fc32(x, N_FREQ));
        CHECK_ESP_DSP(dsps_bit_rev2r_fc32(x, N_FREQ));
    }
    return ESP_OK;
}

/**
 * Compute the FFT on the 2*N_FREQ-length real input signal, overwriting it
 * with an N_FREQ-length complex signal.
 */
esp_err_t OPTIMIZE_FOR_SPEED rfft(float* x) {
    CHECK_ESP_DSP(fft_core(x));
    CHECK_ESP_DSP(dsps_cplx2real_fc32(x, N_FREQ));
    return ESP_OK;
}

/**
 * Compute the inverse FFT on the N_FREQ-length complex input signal,
 * overwriting it with a 2*N_FREQ-length real signal. The output is scaled by
 * 1/N_FREQ.
 */
esp_err_t OPTIMIZE_FOR_SPEED irfft(float* x) {
    // Prepare for IFFT
    float re = x[0], im = x[1];
    array_conj(x, N_FREQ);
    CHECK_ESP_DSP(dsps_cplx2real_fc32(x, N_FREQ));
    x[0] = (re + im) * 0.5;
    x[1] = (im - re) * 0.5;

    // Perform FFT
    CHECK_ESP_DSP(fft_core(x));

    // Scale the output
    array_conj(x, N_FREQ);
    float scale = 1.0 / N_FREQ;
    for (int i = 0; i < N_FREQ*2; i++) { x[i] *= scale; }  // TODO: use dsps_mulc_f32(x, x, N_FREQ*2, scale, 1, 1)

    return ESP_OK;
}

/**
 * Copy the FFT output to the STFT output.
 * This involves a transpose.
 */
void OPTIMIZE_FOR_SPEED copy_fft_to_stft_out(float* fft, float* out) {
    for (int k = 1; k < N_FREQ; k++) {
        out[(k-1)*N_TIME*2] = fft[2*k];
        out[(k-1)*N_TIME*2 + 1] = fft[2*k+1];
    }
    // k == N_FREQ - 1 comes from the second element of the FFT data
    out[(N_FREQ-1)*N_TIME*2] = fft[1];
    out[(N_FREQ-1)*N_TIME*2 + 1] = 0;
}

/**
 * Compute the Short-Time Fourier Transform (STFT) for a single channel input
 * signal. See `compute_spectrogram()` for more details (which works with
 * multiple channels).
 */
void OPTIMIZE_FOR_SPEED compute_stft(
    const float* const x,       // in, length of N_SAMPLES
    const int first_window,     // first time slice index to compute
    float* out                  // out, shape of (N_FREQ, N_TIME)*2 [actually cfloat of size (N_FREQ, N_TIME)]
) {
    float temp[WINDOW_SIZE];
    int start = first_window;

    // first time slice (zero-padded on the left)
    if (first_window == 0) {
        memset(temp, 0, WINDOW_SIZE_HALF * sizeof(float));                  // TODO: use dsps_memset(...) [only optimized on ESP32-S3]
        //for (int k = 0; k < WINDOW_SIZE_HALF; k++) { out[k+WINDOW_SIZE_HALF] = x[k] * WINDOW[k+WINDOW_SIZE_HALF]; }
        dsps_mul_f32(x, WINDOW+WINDOW_SIZE_HALF, temp+WINDOW_SIZE_HALF, WINDOW_SIZE_HALF, 1, 1, 1);
        rfft(temp);
        copy_fft_to_stft_out(temp, out);
        start = 1;
    }

    // middle time slices
    for (int j = start; j < N_TIME - 1; j++) {
        //for (int k = 0; k < WINDOW_SIZE; k++) { temp[k] = xx[k] * WINDOW[k]; }
        dsps_mul_f32(&x[(j-1)*WINDOW_SIZE_HALF], WINDOW, temp, WINDOW_SIZE, 1, 1, 1);
        rfft(temp);
        copy_fft_to_stft_out(temp, &out[j*2]);
    }

    // last time slice (zero-padded on the right)
    //for (int k = 0; k < WINDOW_SIZE_HALF; k++) { temp[k] = xx[k] * WINDOW[k]; }
    dsps_mul_f32(&x[(N_TIME-2)*WINDOW_SIZE_HALF], WINDOW, temp, WINDOW_SIZE_HALF, 1, 1, 1);
    memset(temp+WINDOW_SIZE_HALF, 0, WINDOW_SIZE_HALF * sizeof(float));      // TODO: use dsps_memset(...) [only optimized on ESP32-S3]
    rfft(temp);
    copy_fft_to_stft_out(temp, &out[(N_TIME-1)*2]);
}



void OPTIMIZE_FOR_SPEED compute_stft_quick(
    const float* const x,       // in, length of N_SAMPLES
    float* out                  // out, shape of (N_FREQ, N_TIME)*2 [actually cfloat of size (N_FREQ, N_TIME)]
) {
    float temp[WINDOW_SIZE];

    // second-to-last time slice
    //for (int k = 0; k < WINDOW_SIZE; k++) { temp[k] = xx[k] * WINDOW[k]; }
    dsps_mul_f32(&x[0], WINDOW, temp, WINDOW_SIZE, 1, 1, 1);
    rfft(temp);
    copy_fft_to_stft_out(temp, &out[(N_TIME-2)*2]);

    // last time slice (zero-padded on the right)
    //for (int k = 0; k < WINDOW_SIZE_HALF; k++) { temp[k] = xx[k] * WINDOW[k]; }
    dsps_mul_f32(&x[WINDOW_SIZE_HALF], WINDOW, temp, WINDOW_SIZE_HALF, 1, 1, 1);
    memset(temp+WINDOW_SIZE_HALF, 0, WINDOW_SIZE_HALF * sizeof(float));      // TODO: use dsps_memset(...) [only optimized on ESP32-S3]
    rfft(temp);
    copy_fft_to_stft_out(temp, &out[(N_TIME-1)*2]);
}

/**
 * Construct the two-dimensional weighted spectrogram histogram for each
 * channel. Step 1 of the DUET algorithm.
 *
 * This computes the Short-Time Fourier Transform (STFT) for each of the
 * channels the input signal `x`. The first and last time slices will be
 * zero-padded as appropriate.
 * 
 * This only computes the values for the newest time slices (pass N_TIME to get
 * all time slices). This assumes that the `out` array is already rolled. If
 * a value less than N_TIME is passed then the first time slice won't be
 * zero-padded.
 * 
 * Before calling this function, you must call:
 *   - `init_stft_window()` to initialize the window coefficients.
 *   - `init_stft_fft()` to initialize the FFT library.
 */
void OPTIMIZE_FOR_SPEED compute_spectrogram(
    const float* const x,   // in, shape of (N_CHANNELS, N_SAMPLES)
    int new_times,
    cfloat* out             // out, shape of (N_CHANNELS, N_FREQ, N_TIME)
) {
    for (int i = 0; i < N_CHANNELS; i++) {
        compute_stft(&x[i*N_SAMPLES], N_TIME - new_times, (float*)&out[i*N_FREQ_TIME]);
        //compute_stft_quick(&x[i*N_SAMPLES], (float*)&out[i*N_FREQ_TIME]);
    }
}


void OPTIMIZE_FOR_SPEED compute_spectrogram_alt(
    float** const x,  // in, shape of (N_CHANNELS, N_SAMPLES)
    int new_times,
    cfloat* out             // out, shape of (N_CHANNELS, N_FREQ, N_TIME)
) {
    for (int i = 0; i < N_CHANNELS; i++) {
        compute_stft(x[i], N_TIME - new_times, (float*)&out[i*N_FREQ_TIME]);
        //compute_stft_quick(x[i], (float*)&out[i*N_FREQ_TIME]);
    }
}


/////////////////////////////////////////////////
///////// Compute Attenuation and Delay /////////
/////////////////////////////////////////////////

/** Initialize the global inverse of the frequency values. */
void init_freqs_inv() {
    for (int i = 0; i < N_FREQ; i++) { FREQS_INV[i] = recip(FREQUENCIES[i]); }
}

/**
 * Compute the relative symmetric attenuation (alpha) and delay (delta) for
 * each value in the spectrogram. This gives us phase and amplitude of the left
 * and right channels. Step 2 of the DUET algorithm.
 * 
 * This function only works with two channels, to work with more than two
 * channels, use the `compute_atten_and_delay()` function.
 * 
 * This only computes the values for the newest time slices (pass N_TIME to get
 * all time slices). This assumes that the `alpha` and `delta` arrays are
 * already rolled.
 * 
 * Requires `init_freqs_inv()` to be called before this function.
 */
inline void OPTIMIZE_FOR_SPEED compute_atten_and_delay_2(
    const cfloat * const spectrogram, // in, shape (N_FREQ, N_TIME)
    const int new_times,
    float* alpha,                     // out, shape (N_FREQ, N_TIME)
    float* delta                      // out, shape (N_FREQ, N_TIME)
) {
    const cfloat * const spec0 = spectrogram;  // 5.011 ms, 0.788 ms        4.950 ms, 0.774 ms
    const cfloat * const spec1 = &spectrogram[N_FREQ_TIME];
    int old_times = N_TIME - new_times;
    for (int f = 0, i = old_times; f < N_FREQ; f++, i += old_times) {
        float freq_inv = FREQS_INV[f];
        for (int i_end = i + new_times; i < i_end; i++) {
            cfloat lr_ratio = (spec1[i] + FLT_EPSILON) / (spec0[i] + FLT_EPSILON);
            // Note: using this instead of the easy formula is actually slower but does at least compute the correct values
            // cfloat lr_ratio; // z1/z2 = (a+ib)/(c+id) = (a*c + b*d + i * (b*c - a*d)) / (c*c + d*d)
            // {
            //     float a = creal(spec1[i]) + FLT_EPSILON, b = cimag(spec1[i]);
            //     float c = creal(spec0[i]) + FLT_EPSILON, d = cimag(spec0[i]);
            //     float numer_real = a*c + b*d, numer_imag = b*c - a*d, denom = recip(c*c + d*d);
            //     lr_ratio = (numer_real * denom + numer_imag * denom * I);
            // }

            //float a = cabsf(lr_ratio);
            //alpha[i] = a - 1/a; // => (a^2 - 1) / a
            float a2 = cabs2(lr_ratio);
            alpha[i] = (a2 - 1) * recip_sqrt_fast(a2);  // max ±0.015% error (most have no error); ~1.5x faster

            // float x = cargf(lr_ratio);
            // float y = carg_fast(lr_ratio);
            // float x_y = x / y, y_x = y / x;

            //delta[i] = -cargf(lr_ratio) * freq_inv;
            delta[i] = -carg_fast(lr_ratio) * freq_inv;  // between 0.08% lower to 0.02% higher value (avg 0.008% higher); 1.5x faster
            // TODO: for the highest frequency, this ends up producing several -1 instead of +1 (but it is cyclic so technically correct)

            // TODO: implement big-delay correction (just a 3x3 mean filter on delta?) Section 8.4 in the paper.
            // Could maybe use dspi_conv_f32() but that isn't optimized for any chip and doesn't take advantage of the fact that the filter is separable.
            // Maybe dsps_conv_f32_ae32() would work which is optimized but what are the edge conditions? And maybe just implement a simple 3x3 mean filter
            // would work best (since we could turn every 3 multiplications and 3 additions into 1 multiplication and 3 additions).
        }
    }
}

/**
 * Compute the relative symmetric attenuation (alpha) and delay (delta) for
 * each value in the spectrogram. This gives us phase and amplitude of the left
 * and right channels. Step 2 of the DUET algorithm.
 * 
 * This function works with any number of channels (>=2) by calling the
 * `compute_atten_and_delay_2()` function for each pair of neighboring channels
 * based on:
 *   Speech Separation with Microphone Arrays using the Mean Shift Algorithm
 *   (Ayllón, Gil-Pita, Manuel Rosa-Zurera, 2012)
 *
 * This only computes the values for the newest time slices (pass N_TIME to get
 * all time slices). This assumes that the `alpha` and `delta` arrays are
 * already rolled.
 * 
 * Requires `init_freqs_inv()` to be called before this function.
 */
void OPTIMIZE_FOR_SPEED compute_atten_and_delay(
    const cfloat * const spectrogram, // in, shape (N_CHANNELS, N_FREQ, N_TIME)
    const int new_times,
    float* alpha,                     // out, shape (N_CHANNELS-1, N_FREQ, N_TIME)
    float* delta                      // out, shape (N_CHANNELS-1, N_FREQ, N_TIME)
) {
    for (int i = 0; i < N_CHANNELS-1; i++) {
        compute_atten_and_delay_2(
            &spectrogram[i*N_FREQ_TIME], new_times, &alpha[i*N_FREQ_TIME], &delta[i*N_FREQ_TIME]
        );
    }
}


///////////////////////////////////
///////// Compute Weights /////////
///////////////////////////////////

/** Initialize the global frequencies raised to the power of Q. */
void init_freqs_pow_q() {
    if (Q == 0.0f) {
        for (int i = 0; i < N_FREQ; i++) { FREQS_POW_Q[i] = 1.0f; }
    } else {
        for (int i = 0; i < N_FREQ; i++) { FREQS_POW_Q[i] = powf(fabsf(FREQUENCIES[i]), Q); }
    }
}

/**
 * Compute the weights for every point in the spectrogram. First part of
 * step 3 of the DUET algorithm.
 *
 * This function only works with two channels, to work with more than two
 * channels, use the `compute_atten_and_delay()` function.
 *
 * This only computes the values for the newest time slices (pass N_TIME to get
 * all time slices). This assumes that the `tf_weights` array is already rolled.
 * 
 * Uses the global P and Q values to compute the weights. Requires
 * `init_freqs_pow_q()` to be called before this function if Q is non-zero.
 */
void compute_weights_2(  // NOTE: putting OPTIMIZE_FOR_SPEED on this function causes it to slow down by a lot
    const cfloat * const spectrogram, // in, shape (N_FREQ, N_TIME)
    const int new_times,
    float* tf_weights                 // out, shape (N_FREQ, N_TIME)
) {
    const cfloat * const spec0 = spectrogram;
    const cfloat * const spec1 = &spectrogram[N_FREQ_TIME];
    int old_times = N_TIME - new_times;
    int i = old_times;
    float freq_pow_q = 1.0f;
    for (int f = 0; f < N_FREQ; f++, i += old_times) {
        if (Q != 0.0f) { freq_pow_q = FREQS_POW_Q[f]; }
        for (int i_end = i + new_times; i < i_end; i++) {
            float tf_weight_val = sqrt_fast(cabs2(spec0[i]) * cabs2(spec1[i]));
            if (P != 1.0f) { tf_weight_val = powf(tf_weight_val, P); }
            if (Q != 0.0f) { tf_weight_val *= freq_pow_q; }  // FREQS_POW_Q[i / N_TIME]
            tf_weights[i] = tf_weight_val;
        }
    }
}

/**
 * Compute the weights for every point in the spectrogram. First part of
 * step 3 of the DUET algorithm.
 *
 * This function works with any number of channels (>=2) by calling the
 * `compute_weights_2()` function for each pair of neighboring channels.
 *
 * This only computes the values for the newest time slices (pass N_TIME to get
 * all time slices). This assumes that the `tf_weights` array is already rolled.
 *
 * Uses the global P and Q values to compute the weights. Requires
 * `init_freqs_pow_q()` to be called before this function if Q is non-zero.
 */
void OPTIMIZE_FOR_SPEED compute_weights(
    const cfloat * const spectrogram, // in, shape (N_CHANNELS, N_FREQ, N_TIME)
    const int new_times,
    float* tf_weights                 // out, shape (N_CHANNELS-1, N_FREQ, N_TIME)
) {
    for (int i = 0; i < N_CHANNELS-1; i++) {
        compute_weights_2(
            &spectrogram[i*N_FREQ_TIME], new_times, &tf_weights[i*N_FREQ_TIME]
        );
    }
}


//////////////////////////////
///////// Find Peaks /////////
//////////////////////////////

/** Initialize the temporary vectors used for finding peaks. */
static void init_find_peaks() {
    ms_points.reserve(N_FREQ_TIME/4);
    ms_weights.reserve(N_FREQ_TIME/4);
    ms_centroids.reserve(16);
}

/** Deinitialize (shrink) the temporary vectors used for finding peaks. */
static void deinit_find_peaks() {
    ms_points.clear();
    ms_points.shrink_to_fit();
    ms_weights.clear();
    ms_weights.shrink_to_fit();
    ms_centroids.clear();
    ms_centroids.shrink_to_fit();
}

/**
 * Get the mean-shift points from the spectrogram data.
 * This extracts the points from the spectrogram data that have a weight above
 * the POINT_THRESHOLD. It also checks that the alpha and delta values are
 * within the bounds of ATTENUATION_MAX and DELAY_MAX. The points are stored
 * in the `points` vector and the weights in the `weights` vector.
 */
static void get_ms_points(
    const float * const tf_weights, // in, shape (N_CHANNELS-1, N_FREQ, N_TIME)
    const float * const alpha,      // in, shape (N_CHANNELS-1, N_FREQ, N_TIME)
    const float * const delta,      // in, shape (N_CHANNELS-1, N_FREQ, N_TIME)
    std::vector<DuetMeanShift::point_t>& points,    // out, length n_pts
    std::vector<float>& weights                     // out, length n_pts
) {
    for (int i = 0; i < N_FREQ_TIME; i++) {
        if (tf_weights[i] <= POINT_THRESHOLD) { continue; }

        // NOTE: this alternates alphas and deltas, the original Python code instead places all
        // alphas first and then all deltas. Only changes how mean-shift works, not the results.
        const float * const alpha_i = &alpha[i];
        const float * const delta_i = &delta[i];
        DuetMeanShift::point_t point;
        for (int c = 0; c < N_CHANNELS-1; c++) {
            float a = alpha_i[c*N_FREQ_TIME];
            float d = delta_i[c*N_FREQ_TIME];
            // Make sure the values are within the bounds
            if (fabsf(a) > ATTENUATION_MAX || fabsf(d) > DELAY_MAX) { goto outer_continue; }
            point[c*2] = a;
            point[c*2+1] = d;
        }
        points.emplace_back(point);
        weights.push_back(tf_weights[i]);
        outer_continue:;
    }
}

/**
 * Find the peaks in the spectrogram data.
 */
void find_peaks(
    const float * const tf_weights, // in, shape (N_CHANNELS-1, N_FREQ, N_TIME)
    const float * const alpha,      // in, shape (N_CHANNELS-1, N_FREQ, N_TIME)
    const float * const delta,      // in, shape (N_CHANNELS-1, N_FREQ, N_TIME)
    std::vector<float>& alpha_peaks, // out, shape (n_sources, N_CHANNELS-1)
    std::vector<float>& delta_peaks  // out, shape (n_sources, N_CHANNELS-1)
) {
    // clear temporary vectors
    ms_points.clear();
    ms_weights.clear();
    ms_centroids.clear();

    get_ms_points(tf_weights, alpha, delta, ms_points, ms_weights);
    mean_shift.compute_seeds(ms_points, ms_centroids);
    if (ms_centroids.empty()) { return; }
    mean_shift.mean_shift(ms_points, ms_weights, ms_centroids);

    // clear the output vectors
    alpha_peaks.clear();
    delta_peaks.clear();
    alpha_peaks.reserve(ms_centroids.size() * (N_CHANNELS-1));
    delta_peaks.reserve(ms_centroids.size() * (N_CHANNELS-1));

    for (int i = 0; i < ms_centroids.size(); i++) {
        const DuetMeanShift::point_t& centroid_i = ms_centroids[i];
        for (int c = 0; c < N_CHANNELS-1; c++) {
            alpha_peaks.push_back(centroid_i[c*2]);
            delta_peaks.push_back(centroid_i[c*2+1]);
        }
    }
}


/////////////////////////////////////////////////
///////// Convert Symmetric Attenuation /////////
/////////////////////////////////////////////////

/** Convert the symmetric attenuation values to attenuation values in-place. */
void convert_sym_to_atn(std::vector<float>& atn) {
    for (float& a : atn) { a = 0.5f * (a + sqrt_fast(a * a + 4)); }
}


/////////////////////////
///////// Demix /////////
/////////////////////////
/**
 * Full Demixing when there is only one source. Turns the binaural spectrogram
 * into a monaural one. The demixed output has the DC component skipped.
 */
void full_demix_1(
    const cfloat * const spectrogram, // in, shape (2, N_FREQ, N_TIME)
    const float alpha,                // in
    const float delta,                // in
    std::vector<cfloat> &demixed,     // out, shape (1, N_FREQ, N_TIME)
    uint8_t* best                     // out, shape (N_FREQ, N_TIME)
) {
    memset(best, 0, N_FREQ_TIME);  // source 0 is always the best source in this case

    // fill in the rest with the best source
    const cfloat * const spec0 = &spectrogram[0 * N_FREQ_TIME];
    const cfloat * const spec1 = &spectrogram[1 * N_FREQ_TIME];

    // precompute denominator
    float denom = recip(1.0 + alpha * alpha);

    for (int f = 0; f < N_FREQ; f++) {
        // precompute the core for this frequency (not dependent on time)
        float freq = FREQUENCIES[f];
        cfloat core = alpha * iexp_fast(-delta * freq);
        for (int t = 0; t < N_TIME; t++) {
            cfloat spec0_ft = spec0[f*N_TIME+t];
            cfloat spec1_ft = spec1[f*N_TIME+t];
            demixed[f*N_TIME+t] = (core * spec1_ft + spec0_ft) * denom;
        }
    }
}


/**
 * Full Demixing - combines the computation of the best sources with demixing.
 * This is more memory and computationally efficient than first computing the
 * sources and then demixing but it doesn't allow for the reuse of the sources
 * in other computations.
 * 
 * All of steps 5 and 6 of the DUET algorithm are done in this function.
 */
void full_demix(
    const cfloat * const spectrogram, // in, shape (2, N_FREQ, N_TIME)
    const std::vector<float> &alpha,  // in, shape (n_sources, N_CHANNELS-1)
    const std::vector<float> &delta,  // in, shape (n_sources, N_CHANNELS-1)
    std::vector<cfloat> &demixed,     // out, shape (n_sources, N_FREQ, N_TIME)
    uint8_t* best                     // out, shape (N_FREQ, N_TIME)
) {
    // TODO: support >2 channels
    int n_sources = alpha.size() / (N_CHANNELS-1);
    assert(n_sources >= 1 && n_sources <= 255);

    // fill in all demixed values with zeros (so masked sources are 0)
    demixed.clear();
    demixed.resize(n_sources * N_FREQ_TIME);
    memset(demixed.data(), 0, sizeof(cfloat) * n_sources * N_FREQ_TIME);  // TODO: use DSP memset

    if (n_sources == 1) { // special case for one source
        full_demix_1(spectrogram, alpha[0], delta[0], demixed, best);
        return;
    }

    // fill in the rest with the best source
    const cfloat * const spec0 = &spectrogram[0 * N_FREQ_TIME];
    const cfloat * const spec1 = &spectrogram[1 * N_FREQ_TIME];

    // precompute denominators: 1 / (1 + alphas^2)
    float denom[n_sources];
    for (int s = 0; s < n_sources; s++) { denom[s] = recip(1.0 + alpha[s] * alpha[s]); }

    for (int f = 0; f < N_FREQ; f++) {
        // precompute the core for this frequency (not dependent on time)
        float freq = FREQUENCIES[f];
        cfloat core[n_sources];
        // Using iexp_fast() instead of cexpf() results in ±0.05% relative error and +0.0004
        // absolute error in the magnitude of the demixed output. This isn't enough to effect the
        // "best" output in any major way (tests show that the best source is still the same).
        // It increases the speed of the demixing by about 1.5x. If more accuracy is needed, an
        // alternate version of iexp_fast() can be used that uses the built-in sin/cos functions
        // instead of the fast approximations and it is still ~15% faster since it doesn't compute
        // the real part of the exponent. Another option is to increase the accuracy of the sin/cos
        // approximations (add more table entries and/or use higher-resolution table values).
        //for (int s = 0; s < n_sources; s++) { core[s] = alpha[s] * cexpf(-I * delta[s] * freq); }
        for (int s = 0; s < n_sources; s++) { core[s] = alpha[s] * iexp_fast(-delta[s] * freq); }

        for (int t = 0; t < N_TIME; t++) {
            cfloat spec0_ft = spec0[f*N_TIME+t];
            cfloat spec1_ft = spec1[f*N_TIME+t];

            // find the best source for this frequency and time
            float min_score = FLT_MAX;
            uint8_t best_index = 0;
            for (uint8_t s = 0; s < n_sources; s++) {
                float score = cabs2(core[s] * spec0_ft - spec1_ft) * denom[s];
                if (score < min_score) { min_score = score; best_index = s; }
            }

            // store the best source in the output
            cfloat src_val = (core[best_index] * spec1_ft + spec0_ft) * denom[best_index];
            demixed[best_index*N_FREQ_TIME+f*N_TIME+t] = src_val;
            best[f*N_TIME+t] = best_index;
        }
    }
}


/////////////////////////////////////////
///////// Check for Bad Sources /////////
/////////////////////////////////////////

bool is_bad_source(const cfloat* const source) { // in, shape (N_FREQ, N_TIME)
    // TODO
    return false;
}

bool check_for_bad_sources(
    const std::vector<cfloat>& demixed, // in, shape (n_sources, N_FREQ, N_TIME)
    std::vector<bool>& bad              // out, shape (n_sources)
) {
    int n_sources = demixed.size() / (N_FREQ_TIME);
    bad.clear();
    bad.reserve(n_sources); // reserve space for the number of sources
    bool any_bad = false;
    for (int i = 0; i < n_sources; i++) {
        bool is_bad = is_bad_source(&demixed[i*N_FREQ_TIME]);
        bad.push_back(is_bad);
        any_bad |= is_bad;
    }
    return any_bad;
}

void zero_out_bad_sources(
    const std::vector<bool>& bad,   // in, shape (n_sources)
    const uint8_t* const best,      // in, shape (N_FREQ, N_TIME)
    cfloat* spectrogram             // in/out, shape (N_CHANNELS, N_FREQ, N_TIME)
) {
    for (int i = 0; i < N_FREQ_TIME; i++) {
        if (bad[best[i]]) {
            for (int c = 0; c < N_CHANNELS; c++) { spectrogram[c*N_FREQ_TIME + i] = 0; }
        }
    }
}



//////////////////////////////////////////
///////// Convert to Time Domain /////////
//////////////////////////////////////////

// TODO:


///////////////////////////
///////// Overall /////////
///////////////////////////


static float* audio_temp = NULL;    // shape N_CHANNELS, WINDOW_SIZE_HALF*DECIMATION
static float* audio = NULL;         // shape N_CHANNELS, N_SAMPLES
static cfloat* spectrogram = NULL;  // shape N_CHANNELS, N_FREQ, N_TIME
static float* alpha = NULL;         // shape N_CHANNELS-1, N_FREQ, N_TIME
static float* delta = NULL;         // shape N_CHANNELS-1, N_FREQ, N_TIME
static float* weights = NULL;       // shape N_CHANNELS-1, N_FREQ, N_TIME
static std::vector<float> alpha_peaks;         // shape n_sources, N_CHANNELS-1
static std::vector<float> delta_peaks;         // shape n_sources, N_CHANNELS-1
static std::vector<cfloat> demixed_sources;    // shape n_sources, N_FREQ, N_TIME
static uint8_t* best = NULL;        // shape N_FREQ, N_TIME
static std::vector<bool> bad;       // shape n_sources

template <typename T>
static T* malloc0(size_t count) {
    T* ptr = (T*)malloc(count*sizeof(T));
    if (ptr) {
        memset(ptr, 0, count*sizeof(T));
    }
    return ptr;
}

/**
 * Initializes the Duet audio processing library.
 * This function must be called before using any other functions in the library.
 * It initializes the FFT library, precomputes numerous values, and sets up the
 * necessary parameters for the DUET algorithm along with allocating memory for
 * various buffers.
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
) {
    if (weights && best) { return ESP_OK; } // already initialized

#ifdef DUET_OFFLINE_PREPROCESS
    WINDOW_SIZE = window_size;
    N_SAMPLES = n_samples;
    P = p;
    Q = q;
    POINT_THRESHOLD = point_threshold;
    if (WINDOW_SIZE == 0 || (WINDOW_SIZE & (WINDOW_SIZE - 1)) != 0) { return ESP_ERR_INVALID_ARG; }
    if (WINDOW_SIZE < 8 || WINDOW_SIZE > 8192) { return ESP_ERR_INVALID_ARG; }
    if (N_SAMPLES == 0 || (N_SAMPLES % (WINDOW_SIZE / 2)) != 0) { return ESP_ERR_INVALID_ARG; }
    WINDOW_SIZE_HALF = WINDOW_SIZE >> 1;
    HOP = WINDOW_SIZE_HALF;
    IS_POWER_OF_FOUR = (WINDOW_SIZE & 0xAAAAAAAA) == 0;
    USE_RADIX_4 = !IS_POWER_OF_FOUR;
    WINDOW = (float*)memalign(16, WINDOW_SIZE * sizeof(float));
    DUAL_WINDOW = (float*)memalign(16, WINDOW_SIZE * sizeof(float));
    N_FREQ = WINDOW_SIZE_HALF;
    FREQUENCIES = (float*)memalign(16, N_FREQ * sizeof(float));
    N_TIME = N_SAMPLES / WINDOW_SIZE_HALF + 1;
    N_FREQ_TIME = N_FREQ * N_TIME;
    FREQS_INV = (float*)memalign(16, N_FREQ * sizeof(float));
    FREQS_POW_Q = (Q == 0.0f) ? NULL : (float*)memalign(16, N_FREQ * sizeof(float));
    AUDIO_FRAME_INIT_SIZE = WINDOW_SIZE_HALF * DECIMATION;
    if (!WINDOW || !DUAL_WINDOW || !FREQUENCIES || !FREQS_INV || (Q != 0.0f && !FREQS_POW_Q)) { duet_deinit(); return ESP_ERR_NO_MEM; }
#endif

    init_decimate_fir_coeffs();
    for (int i = 0; i < N_CHANNELS; i++) {
        esp_err_t err = init_decimate_filter(&decimation_filters[i]);
        if (err != ESP_OK) { duet_deinit(); return err; }
    }
    init_stft_window();
    init_stft_dual_window();
    esp_err_t err = init_stft_fft();
    if (err != ESP_OK) { duet_deinit(); return err; }
    init_stft_freqs();
    init_freqs_inv();
    if (Q != 0.0f) { init_freqs_pow_q(); }
    init_find_peaks();

    // Allocate memory for the audio buffer and other arrays
    audio_temp = malloc0<float>(N_CHANNELS * WINDOW_SIZE_HALF * DECIMATION);
    audio = malloc0<float>(N_CHANNELS * N_SAMPLES);
    spectrogram = malloc0<cfloat>(N_CHANNELS * N_FREQ_TIME);
    alpha = malloc0<float>((N_CHANNELS-1) * N_FREQ_TIME);
    delta = malloc0<float>((N_CHANNELS-1) * N_FREQ_TIME);
    weights = malloc0<float>((N_CHANNELS-1) * N_FREQ_TIME);
    best = malloc0<uint8_t>(N_FREQ_TIME);
    if (!audio_temp || !audio || !spectrogram || !alpha || !delta || !weights || !best) {
        duet_deinit();
        return ESP_ERR_NO_MEM;
    }

    // Start with 8 sources (can grow more later)
    alpha_peaks.reserve(8*(N_CHANNELS-1));
    delta_peaks.reserve(8*(N_CHANNELS-1));
    demixed_sources.reserve(8*N_FREQ_TIME);
    bad.reserve(8);

    return ESP_OK;
}

/**
 * Deinitialize the DUET algorithm. This frees any resources that were
 * allocated by duet_init().
 */
void duet_deinit() {
    free(audio_temp); audio_temp = NULL;
    free(audio); audio = NULL;
    free(spectrogram); spectrogram = NULL;
    free(alpha); alpha = NULL;
    free(delta); delta = NULL;
    free(weights); weights = NULL;
    free(best); best = NULL;
    alpha_peaks.clear();
    alpha_peaks.shrink_to_fit();
    delta_peaks.clear();
    delta_peaks.shrink_to_fit();
    demixed_sources.clear();
    demixed_sources.shrink_to_fit();
    bad.clear();
    bad.shrink_to_fit();
    deinit_find_peaks();
    for (int i = 0; i < N_CHANNELS; i++) {
        deinit_decimate_filter(&decimation_filters[i]);
    }
    deinit_stft_fft();
#ifdef DUET_OFFLINE_PREPROCESS
    free(WINDOW); WINDOW = NULL;
    free(DUAL_WINDOW); DUAL_WINDOW = NULL;
    free(FREQUENCIES); FREQUENCIES = NULL;
    free(FREQS_INV); FREQS_INV = NULL;
    free(FREQS_POW_Q); FREQS_POW_Q = NULL;
#endif
}

/**
 * Reset the DUET state for completely new, unrelated, audio. Does not
 * deallocate or deinitialize any buffers or precomputed values.
 */
void duet_reset() {
    // Zero out anything that will be rolled
    memset(audio_temp, 0, N_CHANNELS * WINDOW_SIZE_HALF * DECIMATION * sizeof(float));
    memset(audio, 0, N_CHANNELS * N_SAMPLES * sizeof(float));
    memset(spectrogram, 0, N_CHANNELS * N_FREQ_TIME * sizeof(cfloat));
    memset(alpha, 0, (N_CHANNELS-1) * N_FREQ_TIME * sizeof(float));
    memset(delta, 0, (N_CHANNELS-1) * N_FREQ_TIME * sizeof(float));
    memset(weights, 0, (N_CHANNELS-1) * N_FREQ_TIME * sizeof(float));

    // Clear a few other buffers just in case
    alpha_peaks.clear();
    delta_peaks.clear();
    demixed_sources.clear();
    bad.clear();
}

/**
 * Add the new audio frame to the existing audio buffer and process it with
 * DUET. The new audio frame is interleaved channel data with
 * `AUDIO_FRAME_INIT_SIZE` samples for each channel. This fills the
 * demixed_sources buffer with the demixed sources.
 */
void _process_audio_frame(const float * const frame) {
    // TODO: test with roll, roll2, roll_with_buffer, etc

    // Decimate the new data, placing the results at the end of the audio buffer
    roll2(audio, N_CHANNELS*N_SAMPLES, WINDOW_SIZE_HALF);
    decimate(frame, AUDIO_FRAME_INIT_SIZE, audio, N_SAMPLES - WINDOW_SIZE_HALF);

    // Compute the spectrogram for the new audio data
    roll2(spectrogram, N_CHANNELS*N_FREQ_TIME, 1);
    compute_spectrogram(audio, 2, spectrogram);

    // Compute the alpha, delta, and weights for the new spectrogram
    roll2(alpha, (N_CHANNELS-1)*N_FREQ_TIME, 1);
    roll2(delta, (N_CHANNELS-1)*N_FREQ_TIME, 1);
    roll2(weights, (N_CHANNELS-1)*N_FREQ_TIME, 1);
    compute_atten_and_delay(spectrogram, 2, alpha, delta);
    compute_weights(spectrogram, 2, weights);

    // Find the peaks in the weights, alpha, and delta (i.e. the sources)
    find_peaks(weights, alpha, delta, alpha_peaks, delta_peaks);
    if (alpha_peaks.empty()) { demixed_sources.clear(); return; } // No peaks found, return early

    // Compute the demixed sources based on the peaks
    convert_sym_to_atn(alpha_peaks);
    full_demix(spectrogram, alpha_peaks, delta_peaks, demixed_sources, best);
}

#ifdef DUET_OFFLINE_PREPROCESS
int duet_get_n_channels() { return N_CHANNELS; }
int duet_get_n_freq() { return N_FREQ; }
int duet_get_n_time() { return N_TIME; }
int duet_get_audio_frame_size() { return AUDIO_FRAME_INIT_SIZE; }
const cfloat* duet_process_audio_frame(const float * const frame, int* n_sources) {
    _process_audio_frame(frame);
    *n_sources = demixed_sources.size() / (N_FREQ_TIME);
    return demixed_sources.data();
}
#endif

// void full_process_audio_frame(const int16_t * const frame) {
//     // De-interleave and normalize the new data
//     prep_data(frame, AUDIO_FRAME_INIT_SIZE, audio_temp);

//     // Fill demixed_sources
//     _process_audio_frame(frame);

//     if (demixed_sources.empty()) { return; } // TODO: output silence or the original audio?

//     // Check if any of the sources are bad
//     if (check_for_bad_sources(demixed_sources, bad)) {
//         if (std::all_of(bad.begin(), bad.end(), [](bool b){ return b; })) {
//             // TODO: output silence
//         } else {
//             // Zero out the bad sources
//             zero_out_bad_sources(bad, best, spectrogram);
//             // TODO: convert the spectrogram back to audio
//         }
//     } else {
//         // TODO: output the original audio
//     }
// }
