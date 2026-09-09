#include <stdint.h>
#include <string.h>

#define QK4_0 32

/* fp32 -> fp16, round-half-up. Ties-to-even is skipped deliberately:
 * the scale factor error is second order next to the 4-bit step. */
static unsigned short fp32_to_fp16(float f) {
    unsigned int w;
    memcpy(&w, &f, sizeof(w));
    unsigned int sign = (w >> 16) & 0x8000u;
    int exp = (int)((w >> 23) & 0xffu) - 112;
    unsigned int mant = w & 0x7fffffu;
    if (exp >= 31) {
        return (unsigned short)(sign | 0x7bffu);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (unsigned short)sign;
        }
        mant |= 0x800000u;
        int shift = 14 - exp;
        unsigned int half = (mant + (1u << (shift - 1))) >> shift;
        if (half >= 0x400u) {
            return (unsigned short)(sign | 0x0400u);
        }
        return (unsigned short)(sign | half);
    }
    {
        unsigned int m16 = (mant + 0x1000u) >> 13;
        if (m16 >= 0x400u) {
            m16 = 0;
            exp++;
            if (exp >= 31) {
                return (unsigned short)(sign | 0x7bffu);
            }
        }
        return (unsigned short)(sign | ((unsigned int)exp << 10) | m16);
    }
}

long quantize_row_q4_0(const float *x, void *vy, long k) {
    unsigned char *y;
    long nb;
    long i;
    int j;
    if (!x || !vy || k <= 0 || k % QK4_0 != 0) {
        return -1;
    }
    y = (unsigned char *)vy;
    nb = k / QK4_0;
    for (i = 0; i < nb; i++) {
        float amax = 0.0f;
        float max = 0.0f;
        float d;
        float id;
        unsigned short dh;
        unsigned char *qs;
        for (j = 0; j < QK4_0; j++) {
            float v = x[i * QK4_0 + j];
            float a = v < 0.0f ? -v : v;
            if (a > amax) {
                amax = a;
                max = v;
            }
        }
        d = max / -8.0f;
        id = d != 0.0f ? 1.0f / d : 0.0f;
        dh = fp32_to_fp16(d);
        memcpy(y + i * 18, &dh, 2);
        qs = y + i * 18 + 2;
        for (j = 0; j < QK4_0 / 2; j++) {
            float x0 = x[i * QK4_0 + j] * id + 8.5f;
            float x1 = x[i * QK4_0 + j + QK4_0 / 2] * id + 8.5f;
            int xi0 = (int)x0;
            int xi1 = (int)x1;
            if (xi0 < 0) xi0 = 0; else if (xi0 > 15) xi0 = 15;
            if (xi1 < 0) xi1 = 0; else if (xi1 > 15) xi1 = 15;
            qs[j] = (unsigned char)(xi0 | (xi1 << 4));
        }
    }
    return nb * 18;
}
