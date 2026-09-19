#ifndef EI_MFCC_H
#define EI_MFCC_H

#include <stdint.h>

#define MFCC_SAMPLES 16000

// integer-only mfcc of one 1 s clip at 16 khz, matching the librosa
// reference in data/speech_commands.py. out is 49 x 10, db in q16
void mfcc_q16(const int16_t *pcm, int32_t *out);

// normalize and quantize to the model's int8 input, one requant per
// coefficient: q = requant(mfcc - mean, mult, shift) + zp
void mfcc_to_s8(const int32_t *mfcc, int8_t *out, const int32_t *mean_q16,
                const int32_t *mult, const int8_t *shift, int32_t zp);

#endif
