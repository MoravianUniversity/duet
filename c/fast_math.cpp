#include "fast_math.hpp"
#include <stdint.h>
#include <stdbool.h>

// Most of the functions are small so are included in the header file
// to avoid function call overhead when possible. The larger functions
// are defined in here to avoid bloating the the code too much.



// Sine Table generated with:
//    (np.sin(np.linspace(0, np.pi/2, 256+1)) * (2**24-1)).round().astype(np.int32)
// and then duplicated the last value for overflow protection
constexpr static int32_t __sin_table[] = {
              0,   102943,   205882,   308814,   411733,   514638,
         617523,   720384,   823219,   926022,  1028791,  1131521,
        1234208,  1336849,  1439440,  1541976,  1644455,  1746871,
        1849222,  1951503,  2053710,  2155840,  2257889,  2359854,
        2461729,  2563511,  2665197,  2766783,  2868264,  2969638,
        3070900,  3172046,  3273072,  3373976,  3474752,  3575398,
        3675909,  3776281,  3876512,  3976596,  4076531,  4176312,
        4275936,  4375399,  4474697,  4573827,  4672785,  4771566,
        4870168,  4968587,  5066819,  5164860,  5262706,  5360354,
        5457801,  5555042,  5652074,  5748893,  5845495,  5941878,
        6038036,  6133968,  6229668,  6325134,  6420362,  6515348,
        6610089,  6704581,  6798821,  6892804,  6986528,  7079989,
        7173184,  7266108,  7358759,  7451133,  7543226,  7635035,
        7726557,  7817788,  7908724,  7999363,  8089701,  8179734,
        8269459,  8358873,  8447972,  8536753,  8625212,  8713347,
        8801154,  8888629,  8975770,  9062573,  9149035,  9235152,
        9320921,  9406340,  9491404,  9576111,  9660458,  9744441,
        9828057,  9911303,  9994175, 10076672, 10158789, 10240524,
       10321873, 10402833, 10483402, 10563576, 10643353, 10722728,
       10801700, 10880266, 10958421, 11036164, 11113492, 11190401,
       11266889, 11342953, 11418589, 11493796, 11568570, 11642908,
       11716808, 11790267, 11863282, 11935851, 12007970, 12079637,
       12150849, 12221604, 12291898, 12361730, 12431096, 12499995,
       12568422, 12636377, 12703856, 12770856, 12837376, 12903412,
       12968963, 13034025, 13098596, 13162675, 13226258, 13289342,
       13351927, 13414009, 13475585, 13536655, 13597215, 13657263,
       13716796, 13775813, 13834312, 13892290, 13949744, 14006674,
       14063076, 14118949, 14174290, 14229097, 14283369, 14337103,
       14390297, 14442950, 14495058, 14546621, 14597637, 14648102,
       14698016, 14747377, 14796183, 14844431, 14892121, 14939250,
       14985816, 15031818, 15077255, 15122123, 15166423, 15210151,
       15253307, 15295888, 15337894, 15379322, 15420171, 15460439,
       15500126, 15539228, 15577746, 15615677, 15653021, 15689775,
       15725938, 15761509, 15796487, 15830870, 15864657, 15897847,
       15930438, 15962430, 15993820, 16024609, 16054794, 16084374,
       16113349, 16141718, 16169478, 16196630, 16223172, 16249103,
       16274423, 16299130, 16323223, 16346701, 16369564, 16391811,
       16413441, 16434453, 16454846, 16474619, 16493772, 16512304,
       16530215, 16547503, 16564168, 16580210, 16595627, 16610419,
       16624587, 16638128, 16651043, 16663330, 16674991, 16686024,
       16696428, 16706204, 16715351, 16723868, 16731756, 16739014,
       16745642, 16751639, 16757006, 16761742, 16765847, 16769320,
       16772162, 16774373, 16775952, 16776899, 16777215, 16777215
};
constexpr static int __sin_table_size = (sizeof(__sin_table) / sizeof(__sin_table[0])) - 2; // quarter circle
constexpr static int __sin_table_full = __sin_table_size * 4; // full circle
constexpr static int __sin_table_half = __sin_table_size * 2; // half circle
constexpr static float __sin_table_scale = 1.0f / __sin_table[__sin_table_size];


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
 * 
 * Total this takes ~1 KB of ram for the lookup table.
 */
extern "C" void OPTIMIZE_FOR_SPEED sincos_fast(float radians, float *sine, float *cosine) {
    bool sin_neg = (radians < 0), cos_neg = false;
    if (sin_neg) { radians = -radians; }

    float index = radians * (__sin_table_half / PI_); // convert to an "index" in the lookup table
    long whole = (long)index;
    float remain = index - whole;
    if (whole >= __sin_table_full) { whole %= __sin_table_full; }

    // deal with quadrants
    int_fast16_t y = whole;  // shrink data type since number is always in [0, __sin_table_full)
    if (y >= __sin_table_half) { y -= __sin_table_half; sin_neg = !sin_neg; cos_neg = !cos_neg; }
    if (y >= __sin_table_size) {
        y = __sin_table_half - y;
        if (remain != 0) { remain = 1-remain; y--; }
        cos_neg = !cos_neg;
    }

    // SIN
    int32_t value = __sin_table[y];
    if (likely(remain > 0)) { value += (__sin_table[y+1] - value) * remain; }
    *sine = (sin_neg ? -value : value) * __sin_table_scale;

    // COS
    y = __sin_table_size-y;
    value = __sin_table[y];
    if (likely(remain > 0)) { value += (__sin_table[y-1] - value) * remain; }
    *cosine = (cos_neg ? -value : value) * __sin_table_scale;
}
