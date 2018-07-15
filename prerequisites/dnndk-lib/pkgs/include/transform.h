#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

// transform bgr image:
// b = (b-shiftB)*scaleB
// g = (g-shiftG)*scaleG
// r = (r-shiftR)*scaleR
void transform_bgr(int w, int h, unsigned char *src, signed char *dst,
        float val_shift_B, float var_scale_B,
        float val_shift_G, float var_scale_G,
        float val_shift_R, float var_scale_R);
void transform_bgr_c(int w, int h, unsigned char *src, signed char *dst,
        float val_shift_B, float var_scale_B,
        float val_shift_G, float var_scale_G,
        float val_shift_R, float var_scale_R);
void transform_bgr_intr(int w, int h, unsigned char *src, signed char *dst,
        float var_shift_B, float var_scale_B,
        float var_shift_G, float var_scale_G,
        float var_shift_R, float var_scale_R);

#endif /*__TRANSFORM_H__*/
