#ifndef QUANT_REF_H
#define QUANT_REF_H

/* Reference f32 -> Q4_0 quantizer (port of ggml quantize_row_q4_0_reference).
 * Companion to the dequant reference in dequant_ref.h; first use is the
 * offline Q6_K lm_head -> Q4_0 requant (2x-recipe byte cut).
 * Quantizes k f32 values (k > 0, multiple of 32) into Q4_0 blocks.
 * Returns bytes written (k / 32 * 18), or -1 on bad args.
 */
long quantize_row_q4_0(const float *x, void *y, long k);

#endif
