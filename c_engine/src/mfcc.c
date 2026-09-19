#include "mfcc.h"
#include "mfcc_tables.h"
#include "quantize.h"

// an fft stage can skip halving while every magnitude is under half scale
#define HALF_SCALE_MAG2 (16380u * 16380u)

static int16_t sat16(int32_t x)
{
    return (int16_t)(x > 32767 ? 32767 : x < -32768 ? -32768 : x);
}

static int msb64(uint64_t x)
{
    return 63 - __builtin_clzll(x);
}

// in-place radix-2 decimation-in-time fft on q15 data with block floating
// point: a stage halves its outputs only when a magnitude could overflow
// int16. returns how many stages were halved so the caller can undo it
static int fft_q15(int16_t *re, int16_t *im)
{
    const int n = MFCC_N_FFT;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            int16_t t = re[i];
            re[i] = re[j];
            re[j] = t;
            t = im[i];
            im[i] = im[j];
            im[j] = t;
        }
    }

    int halved = 0;
    for (int half = 1; half < n; half <<= 1) {
        uint32_t peak = 0;
        for (int i = 0; i < n; i++) {
            uint32_t mag2 = (uint32_t)(re[i] * re[i]) + (uint32_t)(im[i] * im[i]);
            if (mag2 > peak)
                peak = mag2;
        }
        int shift = peak > HALF_SCALE_MAG2;
        halved += shift;
        int step = n / (2 * half);
        for (int start = 0; start < n; start += 2 * half)
            for (int j = 0; j < half; j++) {
                int a = start + j, b = a + half;
                int32_t c = mfcc_cos[j * step], s = mfcc_sin[j * step];
                // t = x[b] * e^(-i theta), products fit int32 since |x[b]| <= 32767
                int32_t tr = (c * re[b] + s * im[b] + (1 << 14)) >> 15;
                int32_t ti = (c * im[b] - s * re[b] + (1 << 14)) >> 15;
                int32_t ar = re[a], ai = im[a];
                re[a] = sat16((ar + tr + shift) >> shift);
                im[a] = sat16((ai + ti + shift) >> shift);
                re[b] = sat16((ar - tr + shift) >> shift);
                im[b] = sat16((ai - ti + shift) >> shift);
            }
    }
    return halved;
}

// log2(x) in q16 for x > 0: exponent from the leading bit, the next 8 bits
// index the table, the 8 after that interpolate
static int32_t log2_q16(uint64_t x)
{
    int e = msb64(x);
    uint32_t frac = (uint32_t)(e >= 16 ? x >> (e - 16) : x << (16 - e)) & 0xffff;
    int idx = frac >> 8, rem = frac & 0xff;
    int32_t lo = mfcc_log2_lut[idx], hi = mfcc_log2_lut[idx + 1];
    return e * 65536 + lo + (((hi - lo) * rem + 128) >> 8);
}

// one frame -> 40 log-mel values in q16 db
static void log_mel_frame(const int16_t *x, int32_t *logmel)
{
    static int16_t re[MFCC_N_FFT], im[MFCC_N_FFT];
    static uint32_t power[MFCC_BINS];

    // normalize so the frame peak lands in [2^14, 2^15), tracked as shift s
    int32_t peak = 0;
    for (int i = 0; i < MFCC_N_FFT; i++) {
        int32_t v = x[i] < 0 ? -x[i] : x[i];
        if (v > peak)
            peak = v;
    }
    int s = peak ? 14 - (31 - __builtin_clz((uint32_t)(peak > 32767 ? 32767 : peak))) : 0;
    for (int i = 0; i < MFCC_N_FFT; i++) {
        re[i] = sat16((x[i] * (1 << s) * mfcc_hann[i] + (1 << 14)) >> 15);
        im[i] = 0;
    }

    int g = fft_q15(re, im);
    for (int k = 0; k < MFCC_BINS; k++)
        power[k] = (uint32_t)(re[k] * re[k]) + (uint32_t)(im[k] * im[k]);

    // true power = power * 2^(2g - 30 - 2s) and the weights carry 2^21, so
    // log2(mel) = log2(acc) + 2g - 2s - 51
    const uint16_t *w = mfcc_mel_w;
    for (int m = 0; m < MFCC_MELS; m++) {
        uint64_t acc = 0;
        for (int i = 0; i < mfcc_mel_len[m]; i++)
            acc += (uint64_t)w[i] * power[mfcc_mel_start[m] + i];
        w += mfcc_mel_len[m];
        int32_t db = MFCC_FLOOR_Q16;
        if (acc) {
            int32_t l2 = log2_q16(acc) + (2 * g - 2 * s - 51) * 65536;
            db = (int32_t)(((int64_t)l2 * MFCC_LOG10_2_Q29 + (1 << 28)) >> 29);
        }
        logmel[m] = db < MFCC_FLOOR_Q16 ? MFCC_FLOOR_Q16 : db;
    }
}

void mfcc_q16(const int16_t *pcm, int32_t *out)
{
    int32_t logmel[MFCC_MELS];
    for (int t = 0; t < MFCC_FRAMES; t++) {
        log_mel_frame(pcm + t * MFCC_HOP, logmel);
        for (int k = 0; k < MFCC_COEFFS; k++) {
            int64_t acc = 0;
            for (int m = 0; m < MFCC_MELS; m++)
                acc += (int64_t)mfcc_dct[k * MFCC_MELS + m] * logmel[m];
            out[t * MFCC_COEFFS + k] = (int32_t)((acc + (1 << 29)) >> 30);
        }
    }
}

void mfcc_to_s8(const int32_t *mfcc, int8_t *out, const int32_t *mean_q16,
                const int32_t *mult, const int8_t *shift, int32_t zp)
{
    for (int i = 0; i < MFCC_FRAMES * MFCC_COEFFS; i++) {
        int k = i % MFCC_COEFFS;
        out[i] = clamp_s8(requant(mfcc[i] - mean_q16[k], mult[k], shift[k]) + zp, -128, 127);
    }
}
