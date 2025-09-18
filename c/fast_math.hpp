/**
 * Math utilities for ESP32. These are primarily optimized for speed on the ESP32.
 */

// First some general notes on speeds:
// * See https://www.reddit.com/r/esp32/comments/t46960/whats_the_esp32_s3_floating_point_performance/ for some ESP32-S3 speed tests
//   * `float32` division is abysmal (about 1/50th the speed of `float32` multiplication) but the
//     `recip()` function below can be much faster.
//   * `float32` has roughly the same speed as ints for addition and multiplication but not division
//   * `float32` has special FMA (fused multiply-add) instructions that make it really fast for
//     multiplication and addition together, should use that whenever possible

#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

// Optimize for speed, not size; put in a function definition right before
// function name to apply it to that function. This is more aggression than
// even -O3 and may result in larger code size and inaccurate floating-point
// results. In some cases (due to larger code size), it may even be slower.
// Use with caution.
#ifndef __clang__
#define OPTIMIZE_FOR_SPEED __attribute__((optimize("Ofast")))
#else
#define OPTIMIZE_FOR_SPEED // clang doesn't support this attribute
#endif


// The ESP library defines these macros as no-ops unless all of the entire
// compilation is set to performance (but we want most to be compiled for
// size). So we redefine them here.
#ifdef likely
#undef likely
#endif
#define likely(x)      __builtin_expect(!!(x), 1)
#ifdef unlikely
#undef unlikely
#endif
#define unlikely(x)    __builtin_expect(!!(x), 0)


//////////////////////////////////////////////////
////////// Integer/Float Bit Conversion //////////
//////////////////////////////////////////////////
static inline __attribute__((always_inline)) float OPTIMIZE_FOR_SPEED bits_to_float(int32_t x) {
    union { int32_t i; float f; } u = {x};
    return u.f;
}
// static inline __attribute__((always_inline)) float OPTIMIZE_FOR_SPEED bits_to_float(uint32_t x) {
//     union { uint32_t i; float f; } u = {x};
//     return u.f;
// }
static inline __attribute__((always_inline)) int32_t OPTIMIZE_FOR_SPEED bits_to_int(float x) {
    union { float f; int32_t i; } u = {x};
    return u.i;
}
static inline __attribute__((always_inline)) uint32_t OPTIMIZE_FOR_SPEED bits_to_uint(float x) {
    union { float f; uint32_t i; } u = {x};
    return u.i;
}


/////////////////////////////////////////
////////// Integer Power Check //////////
/////////////////////////////////////////
static inline __attribute__((always_inline)) bool OPTIMIZE_FOR_SPEED is_power_of_two(int x) { return (x != 0) && ((x & (x - 1)) == 0); }  // or dsp_is_power_of_two() from dsp_common.h
static inline __attribute__((always_inline)) bool OPTIMIZE_FOR_SPEED is_power_of_four_given_is_pow_2(int n) { return (n & 0xAAAAAAAA) == 0; }  // assuming n is a power of 2, checks if it is also a power of 4
static inline __attribute__((always_inline)) bool OPTIMIZE_FOR_SPEED is_power_of_four(int n) { return is_power_of_two(n) && is_power_of_four_given_is_pow_2(n); }


//////////////////////////////
////////// Division //////////
//////////////////////////////
#if CONFIG_DSP_OPTIMIZED && CONFIG_IDF_TARGET_ESP32P4 != 1

 /**
  * Calculate the reciprocal of a float using inline assembly.
  * This is MUCH faster than using doing 1.0f/x and fully accurate.
  * From https://github.com/espressif/esp-dsp/issues/95.
  */
static __attribute__((always_inline)) inline
float recip(float x) {
    float result, temp;
    asm(
        "recip0.s %0, %2\n"
        "const.s %1, 1\n"
        "msub.s %1, %2, %0\n"
        "madd.s %0, %0, %1\n"
        "const.s %1, 1\n"
        "msub.s %1, %2, %0\n"
        "maddn.s %0, %0, %1\n"
        :"=&f"(result),"=&f"(temp):"f"(x)
    );
    return result;
}

/**
 * Divide two floats using the reciprocal method.
 * This is faster than using a/b for floats and fully accurate.
 */
#define DIVIDE(a, b) ((a)*recip(b))

#else

// The ESP32-P4 has a 3-cycle fdiv.s instruction which is faster than the above.
// Additionally, the code above may not work on other CPU architectures. In those
// cases, we can use the standard division operator and hope that the compiler
// optimizes it.

static __attribute__((always_inline)) inline
float recip(float x) { return 1.0f / x; }
#define DIVIDE(a, b) ((a) / (b))

#endif


/////////////////////////////////
////////// Square Root //////////
/////////////////////////////////
/**
 * Approximate inverse square root function.
 * This is the `InvSqrt43()` function from https://www.researchgate.net/publication/349173096_SIMPLE_EFFECTIVE_FAST_INVERSE_SQUARE_ROOT_ALGORITHM_WITH_TWO_MAGIC_CONSTANTS
 * This has 21.21 bits of accuracy and costs 281.2 ns on the ESP32 (2.85x faster
 * than `1/sqrt(x)` which has an accuracy of 23.42 bits and costs 801.5 ns).
 * 
 * The `dsps_inverted_sqrtf_f32()` function from `dsps_sqrt.h` has ~9 bits
 * of accuracy and costs ~200ns (5 multiplactions instead of 8).
 */
inline static float OPTIMIZE_FOR_SPEED recip_sqrt_fast(float x) {
    int i = bits_to_int(x);
    int ix = i - 0x80800000;
    i = i >> 1;
    int ii = 0x5E5FB3E2 - i;
    i = 0x5F5FB3E2 - i;
    float y = bits_to_float(i);
    float yy = bits_to_float(ii);
    y = yy*(4.76424932f - x*y*y);
    float mhalf = bits_to_float(ix);
    float t = fmaf(mhalf, y*y, 0.500000298f);
    return fmaf(y, t, y);
}

/**
 * Approximate square root function based on the fact that `sqrt(x) = x * (1/sqrt(x))`
 * and we can compute the inverse square root fast and accurately using the
 * `recip_sqrt_fast()` function.
 * 
 * Compared to sqrtf(x):
 *  - ESP32: ~1.4x faster (266 ns vs 365 ns)
 *  - ESP32-P4: ~31x faster (122 ns vs 3800 ns)  # TODO: why regular is 10x slower on the ESP32-P4? according to https://github.com/espressif/esp-idf/issues/14177#issuecomment-2244149530 it should be only 61 ns
 * It has a max relative error of ~3.58e-5% (14.8 bits of accuracy), and on
 * average it is equal.
 */
inline static float OPTIMIZE_FOR_SPEED sqrt_fast(float x) {
    return x * recip_sqrt_fast(x);

    // `dsps_sqrtf_f32()` version - extremely inaccurate (max 3.5% error
    // [4.8 bits] and 0.2% average error) but very fast (only 106 ns).
    //return bits_to_float(0x1fbb4000 + (bits_to_int(f) >> 1));
}

/* Testing Code
    #define CPU_FREQ 240000.0 // ESP32 CPU frequency in 1/ms; use 360000.0 on ESP32-P4
    #include "duet/fast_math.hpp"

    seed_random();

    esp_cpu_cycle_count_t start, end, diff;

    float* xs = (float*)malloc(10000 * sizeof(float));
    float* ys = (float*)malloc(10000 * sizeof(float));
    for (int i = 0; i < 10000; i++) { xs[i] = random() / (float)RAND_MAX * 200; }

    start = esp_cpu_get_cycle_count();
    for (int i = 0; i < 10000; i++) { ys[i] = sqrt(xs[i]); }
    end = esp_cpu_get_cycle_count();
    diff = end - start;
    printf("sqrt() took %d cycles / %0.3f ms\n", diff, diff / CPU_FREQ);

    start = esp_cpu_get_cycle_count();
    for (int i = 0; i < 10000; i++) { ys[i] = sqrt_fast(xs[i]); }
    end = esp_cpu_get_cycle_count();
    diff = end - start;
    printf("sqrt_fast() took %d cycles / %0.3f ms\n", diff, diff / CPU_FREQ);
    float total_err = 0;
    float max_err = 0;
    for (int i = 0; i < 10000; i++) {
        float err = fabs(ys[i] / sqrt(xs[i]));
        total_err += err;
        if (err > max_err) max_err = err;
    }
    printf("sqrt_fast() max rel error: %0.9f, avg rel error: %0.9f\n", max_err, total_err / 10000.0);
*/


//////////////////////////////
////////// Exponent //////////
//////////////////////////////
/**
 * Fast exponentiation function using a second-order approximation and floating-
 * point tricks. From https://specbranch.com/posts/fast-exp/.
 * 
 * Compared to expf():
 *  - ESP32: ~5.2x faster
 *  - ESP32-P4: ~4.7x faster (107 ns vs 505 ns)
 * Has up to ±0.48% error. It does not maintain monotonicity.
 */
static float OPTIMIZE_FOR_SPEED exp_fast_o2(float x) {
    float xb = x * 12102203;
    int32_t ir = ((int32_t)xb) + (127 << 23) - 345088;
    float first_order = bits_to_float(ir);
    float correction_x = bits_to_float((ir & 0x7fffff) | (127 << 23));
    float correction = fmaf(correction_x, 0.22670517861843109130859375f, -0.671999752521514892578125f);
    correction = fmaf(correction, correction_x, 1.469318866729736328125f);
    return first_order * correction;
}

/**
 * Fast exponentiation function using a first-order approximation and floating-
 * point tricks. From https://specbranch.com/posts/fast-exp/.
 * 
 * Compared to expf():
 *  - ESP32: ~14.7x faster
 *  - ESP32-P4: ~9.1x faster (56 ns vs 505 ns)
 * Has up to ±2.98% error. It does maintain monotonicity.
 */
static float OPTIMIZE_FOR_SPEED exp_fast_o1(float x) {
    float xb = x * 12102203;
    int i = ((int)xb) + 1064986823;
    return bits_to_float(i);
}

/* Testing Code
    #define CPU_FREQ 240000.0 // ESP32 CPU frequency in 1/ms; use 360000.0 on ESP32-P4
    #include "duet/fast_math.hpp"

    seed_random();

    esp_cpu_cycle_count_t start, end, diff;

    float* xs = (float*)malloc(10000 * sizeof(float)); // -6 to 0
    float* ys = (float*)malloc(10000 * sizeof(float));
    for (int i = 0; i < 10000; i++) { xs[i] = random() / (float)RAND_MAX * -6; }

    start = esp_cpu_get_cycle_count();
    for (int i = 0; i < 10000; i++) { ys[i] = expf(xs[i]); }
    end = esp_cpu_get_cycle_count();
    diff = end - start;
    printf("exp() took %d cycles / %0.3f ms\n", diff, diff / CPU_FREQ);

    start = esp_cpu_get_cycle_count();
    for (int i = 0; i < 10000; i++) { ys[i] = exp_fast_o1(xs[i]); }
    end = esp_cpu_get_cycle_count();
    diff = end - start;
    printf("exp_fast_o1() took %d cycles / %0.3f ms\n", diff, diff / CPU_FREQ);

    start = esp_cpu_get_cycle_count();
    for (int i = 0; i < 10000; i++) { ys[i] = exp_fast_o2(xs[i]); }
    end = esp_cpu_get_cycle_count();
    diff = end - start;
    printf("exp_fast_o2() took %d cycles / %0.3f ms\n", diff, diff / CPU_FREQ);
*/


//////////////////////////////////
////////// Trigonometry //////////
//////////////////////////////////
#ifdef __cplusplus
constexpr float PI_ = 3.14159265358979323846f; // pi constant (as a float)
constexpr float PI_HALF = 3.14159265358979323846f / 2; // pi/2 constant (as a float)
constexpr float PI_2 = 3.14159265358979323846f * 2; // 2*pi constant (as a float)
#else
const float PI_ = 3.14159265358979323846f; // pi constant (as a float)
const float PI_HALF = 3.14159265358979323846f / 2; // pi/2 constant (as a float)
const float PI_2 = 3.14159265358979323846f * 2; // 2*pi constant (as a float)
#endif

/**
 * Fast arctan function using a degree-5 approximation. Much faster than the
 * standard library atanf() function but less accurate. Assumes the given value
 * is in the range [-1, 1].
 * From https://github.com/RobTillaart/FastTrig.
 */
static float OPTIMIZE_FOR_SPEED atan_fast_d5(float x) {
    float x2 = x * x;
    return (((0.079331f * x2) - 0.288679f) * x2 + 0.995354f) * x;
}

/**
 * Fast arctan function using a degree-7 approximation. More accurate than the
 * degree-5 approximation but still faster than the standard library atanf()
 * function. Assumes the given value is in the range [-1, 1].
 * From https://github.com/RobTillaart/FastTrig.
 */
static float OPTIMIZE_FOR_SPEED atan_fast_d7(float x) {
    float x2 = x * x;
    return ((((-0.0389929f * x2) + 0.1462766f) * x2 - 0.3211819f) * x2 + 0.9992150f) * x;
}

/**
 * Fast arctan function using a degree-9 approximation. More accurate than the
 * degree-7 approximation but still faster than the standard library atanf()
 * function. Assumes the given value is in the range [-1, 1].
 */
static float OPTIMIZE_FOR_SPEED atan_fast_d9(float x) {
    float x2 = x * x;
    return ((((0.021814232f * x2 - 0.087072968f) * x2 + 0.181389254f) * x2 - 0.330585734f) * x2 + 0.99988258f) * x;
}

/**
 * Same as atan_fast_d5() but does not assume the value is in the range [-1, 1].
 * From https://github.com/RobTillaart/FastTrig.
 */
static float OPTIMIZE_FOR_SPEED atan_fast_d5_ur(float x) {
    if (unlikely(x > +1)) return (+PI_HALF) - atan_fast_d5(recip(x));
    if (unlikely(x < -1)) return (-PI_HALF) - atan_fast_d5(recip(x));
    return atan_fast_d5(x);
}

/**
 * Same as atan_fast_d7() but does not assume the value is in the range [-1, 1].
 * From https://github.com/RobTillaart/FastTrig.
 */
static float OPTIMIZE_FOR_SPEED atan_fast_d7_ur(float x) {
    if (unlikely(x > +1)) return (+PI_HALF) - atan_fast_d7(recip(x));
    if (unlikely(x < -1)) return (-PI_HALF) - atan_fast_d7(recip(x));
    return atan_fast_d7(x);
}

/**
 * Same as atan_fast_d9() but does not assume the value is in the range [-1, 1].
 */
static float OPTIMIZE_FOR_SPEED atan_fast_d9_ur(float x) {
    if (unlikely(x > +1)) return (+PI_HALF) - atan_fast_d9(recip(x));
    if (unlikely(x < -1)) return (-PI_HALF) - atan_fast_d9(recip(x));
    return atan_fast_d9(x);
}

/**
 * Fast 2-argument arctangent function. It uses atan_fast_d5() for the main
 * calculation and adjusts the result based on the signs of x and y.
 * 
 * This is ~3.6x faster (966 vs 4093 cycles, 4 vs 17 us).
 * Max error is ~4.9e-3 radians. Average error of ~3.9e-4 radians.
 */
static float OPTIMIZE_FOR_SPEED atan2_fast_d5(float y, float x) {
    if (unlikely(x == 0)) return y > 0 ? PI_HALF : (y < 0 ? -PI_HALF : NAN);
    if (fabsf(y) >= fabsf(x)) return ((y > 0) ? PI_HALF : -PI_HALF) - atan_fast_d5(DIVIDE(x, y));
    return (x >= 0) ? atan_fast_d5(DIVIDE(y, x)) : (((y > 0) ? PI_: -PI_) + atan_fast_d5(DIVIDE(y, x)));
}

/**
 * Fast 2-argument arctangent function. It uses atan_fast_d7() for the main
 * calculation and adjusts the result based on the signs of x and y.
 *
 * This is ~4.8x faster (678 vs 4093 cycles, 2 vs 17 us).
 * Max error is ~7.5e-4 radians. Average error of ~5.2e-5 radians.
 */
static float OPTIMIZE_FOR_SPEED atan2_fast_d7(float y, float x) {
    if (unlikely(x == 0)) return y > 0 ? PI_HALF : (y < 0 ? -PI_HALF : NAN);
    if (fabsf(y) >= fabsf(x)) return ((y > 0) ? PI_HALF : -PI_HALF) - atan_fast_d7(DIVIDE(x, y));
    return (x >= 0) ? atan_fast_d7(DIVIDE(y, x)) : (((y >= 0) ? PI_: -PI_) + atan_fast_d7(DIVIDE(y, x)));
}

/**
 * Fast 2-argument arctangent function. It uses atan_fast_d9() for the main
 * calculation and adjusts the result based on the signs of x and y.
 *
 * This is ~4.8x faster (1332 vs 4093 cycles, 6 vs 17 us).
 * Max error is ~9.8e-5 radians. Average error of ~6.7e-6 radians.
 */
static float OPTIMIZE_FOR_SPEED atan2_fast_d9(float y, float x) {
    if (unlikely(x == 0)) return y > 0 ? PI_HALF : (y < 0 ? -PI_HALF : NAN);
    if (fabsf(y) >= fabsf(x)) return ((y > 0) ? PI_HALF : -PI_HALF) - atan_fast_d9(DIVIDE(x, y));
    return (x >= 0) ? atan_fast_d9(DIVIDE(y, x)) : (((y > 0) ? PI_: -PI_) + atan_fast_d9(DIVIDE(y, x)));
}

/* Testing Code
    #define CPU_FREQ 240000.0 // ESP32 CPU frequency in 1/ms; use 360000.0 on ESP32-P4
    #include "duet/fast_math.hpp"

    seed_random();

    esp_cpu_cycle_count_t start, end, diff;

    float x = random() / (float)RAND_MAX;
    float y = random() / (float)RAND_MAX;

    start = esp_cpu_get_cycle_count();
    float ans1 = atan2f(y, x); ans1 += atan2f(y, x); ans1 += atan2f(y, x); ans1 += atan2f(y, x); ans1 += atan2f(y, x);
    ans1 += atan2f(y, x); ans1 += atan2f(y, x); ans1 += atan2f(y, x); ans1 += atan2f(y, x); ans1 += atan2f(y, x);
    end = esp_cpu_get_cycle_count();
    diff = (end - start) / 10;
    printf("atan2f: %d cycles / %0.4f ms\n", diff, (diff) / CPU_FREQ);

    start = esp_cpu_get_cycle_count();
    float ans2 = atan2_fast_d5(y, x); ans2 += atan2_fast_d5(y, x); ans2 += atan2_fast_d5(y, x); ans2 += atan2_fast_d5(y, x); ans2 += atan2_fast_d5(y, x);
    ans2 += atan2_fast_d5(y, x); ans2 += atan2_fast_d5(y, x); ans2 += atan2_fast_d5(y, x); ans2 += atan2_fast_d5(y, x); ans2 += atan2_fast_d5(y, x);
    end = esp_cpu_get_cycle_count();
    diff = (end - start) / 10;
    printf("atan2_fast_d5: %d cycles / %0.4f ms\n", diff, (diff) / CPU_FREQ);

    start = esp_cpu_get_cycle_count();
    float ans3 = atan2_fast_d7(y, x); ans3 += atan2_fast_d7(y, x); ans3 += atan2_fast_d7(y, x); ans3 += atan2_fast_d7(y, x); ans3 += atan2_fast_d7(y, x);
    ans3 += atan2_fast_d7(y, x); ans3 += atan2_fast_d7(y, x); ans3 += atan2_fast_d7(y, x); ans3 += atan2_fast_d7(y, x); ans3 += atan2_fast_d7(y, x);
    end = esp_cpu_get_cycle_count();
    diff = (end - start) / 10;
    printf("atan2_fast_d7: %d cycles / %0.4f ms\n", diff, (diff) / CPU_FREQ);

    start = esp_cpu_get_cycle_count();
    float ans4 = atan2_fast_d9(y, x); ans4 += atan2_fast_d9(y, x); ans4 += atan2_fast_d9(y, x); ans4 += atan2_fast_d9(y, x); ans4 += atan2_fast_d9(y, x);
    ans4 += atan2_fast_d9(y, x); ans4 += atan2_fast_d9(y, x); ans4 += atan2_fast_d9(y, x); ans4 += atan2_fast_d9(y, x); ans4 += atan2_fast_d9(y, x);
    end = esp_cpu_get_cycle_count();
    diff = (end - start) / 10;
    printf("atan2_fast_d9: %d cycles / %0.4f ms\n", diff, (diff) / CPU_FREQ);

    printf("atan2f: %f, atan2_fast_d5: %f, atan2_fast_d7: %f, atan2_fast_d9: %f\n", ans1/10, ans2/10, ans3/10, ans4/10);

    float max_d5_err = fabsf(ans1 - ans2);
    float max_d7_err = fabsf(ans1 - ans3);
    float max_d9_err = fabsf(ans1 - ans4);
    float sum_d5_err = 0.0;
    float sum_d7_err = 0.0;
    float sum_d9_err = 0.0;
    for (int i = 0; i < 1000000; i++) {
        x = random() / (float)RAND_MAX;
        y = random() / (float)RAND_MAX;
        ans1 = atan2f(y, x);
        ans2 = atan2_fast_d5(y, x);
        ans3 = atan2_fast_d7(y, x);
        ans4 = atan2_fast_d9(y, x);
        max_d5_err = fmaxf(max_d5_err, fabsf(ans1 - ans2));
        max_d7_err = fmaxf(max_d7_err, fabsf(ans1 - ans3));
        max_d9_err = fmaxf(max_d9_err, fabsf(ans1 - ans4));
        sum_d5_err += fabsf(ans1 - ans2);
        sum_d7_err += fabsf(ans1 - ans3);
        sum_d9_err += fabsf(ans1 - ans4);
    }

    printf("Max error for atan2_fast_d5: %.9f, atan2_fast_d7: %.9f, atan2_fast_d9: %.9f\n", max_d5_err, max_d7_err, max_d9_err);
    printf("Average error for atan2_fast_d5: %.9f, atan2_fast_d7: %.9f, atan2_fast_d9: %.9f\n", sum_d5_err / 1000000.0, sum_d7_err / 1000000.0, sum_d9_err / 1000000.0);
 */

/**
 * Fast sine and cosine. This uses a precomputed 24-bit sine table for the first
 * quarter of a circle.
 * 
 * Inspired by https://github.com/RobTillaart/FastTrig but even more optimized:
 *   - Uses a 24-bit table instead of 16-bit for better accuracy at effectively
 *     the same speed (but more memory).
 *   - More entries in the table for better accuracy at effectively the same
 *     speed (but more memory).
 *   - Table size is a power of 2 for faster indexing/math.
 *   - Some simplifications taken in the math to increase its speed.
 */
#ifdef __cplusplus
extern "C" {
#endif
void OPTIMIZE_FOR_SPEED sincos_fast(float radians, float *sine, float *cosine);
#ifdef __cplusplus
}
#endif


/////////////////////////////////////
////////// Complex Numbers //////////
/////////////////////////////////////
#ifdef __cplusplus
#include <complex>
typedef std::complex<float> cfloat;
#else
#include <complex.h>  // not C++ compatible in theory
typedef float _Complex cfloat;
#endif

/** Computes cabsf(x)^2, avoiding the sqrtf call. */
static inline __attribute__((always_inline)) OPTIMIZE_FOR_SPEED float cabs2(cfloat x) {
    #ifdef __cplusplus
    return std::norm(x);
    #else
    return crealf(x) * crealf(x) + cimagf(x) * cimagf(x);
    #endif
}

/** Computes cargf(x) using fast atan2(). */
// TODO: tried d9 version but it was slower than using atanf() directly here?
static inline __attribute__((always_inline)) OPTIMIZE_FOR_SPEED float carg_fast(cfloat x) {
    #ifdef __cplusplus
    return atan2_fast_d7(std::imag(x), std::real(x));
    #else
    return atan2_fast_d7(cimagf(x), crealf(x));
    #endif
}

/**
 * Computes the imaginary exponential of x using fast trigronometry functions:
 *       iexp_fast(x) = cos(x) + i*sin(x)
 * See cexp_fast() for using any complex number.
 */
static inline __attribute__((always_inline)) OPTIMIZE_FOR_SPEED cfloat iexp_fast(float x) {
    //return cos(x) + I * sin(x);
    float real, imag;
    sincos_fast(x, &imag, &real);
    #ifdef __cplusplus
    return cfloat(real, imag);
    #else
    return real + I * imag;
    #endif
}

/**
 * Computes the complex exponential of x using fast trigonometry and exp():
 *       cexp_fast(x) = exp(real(x)) * iexp_fast(imag(x))
 * This is faster than using the standard cexpf() function. This uses the
 * order-1 exp() function.
 */
//static inline __attribute__((always_inline)) OPTIMIZE_FOR_SPEED cfloat cexp_fast(cfloat x) {
//    return exp_fast_o1(crealf(x)) * iexp_fast(cimagf(x));
//}
