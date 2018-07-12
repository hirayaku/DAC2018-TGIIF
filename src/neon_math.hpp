#include <cstdint>
#include <arm_neon.h>

using std::exp;
typedef float32x4_t v4sf;
typedef uint32x4_t v4su;
v4sf exp_ps(v4sf);

void softmax_c(const int8_t* input, float scale, unsigned int cls,
    float* output) {
  float sum = 0.f;
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] = exp(input[i] * scale);
    sum += output[i];
  }

  for (unsigned int i = 0; i < cls; ++i) output[i] /= sum;
}

void softmax_c(const int8_t* input, float scale, unsigned int cls,
    unsigned int group, float* output) {
  for (unsigned int i = 0; i < group; ++i) {
    softmax_c(input, scale, cls, output);
    input += cls;
    output += cls;
  }  
}

void softmax4_internal(const int8_t*, float, unsigned int, float*);
void softmax4_neon(const int8_t* input, float scale,
    unsigned int group, float* output) {
  unsigned int aligned = group & (-8);
  softmax4_internal(input, scale, aligned, output);
  unsigned int remain = group - aligned;
  input += (4 * aligned);
  output += (4 * aligned);
  softmax_c(input, scale, 4, remain, output);
}

/*
 * 2-class softmax
 */
void softmax2_internal(const int8_t*, float, unsigned int, float*);
void softmax2_neon(const int8_t* input, float scale,
    unsigned int group, float* output) {
  unsigned int aligned = group & (-8);
  softmax2_internal(input, scale, aligned, output);
  unsigned int remain = group - aligned;
  input += (2 * aligned);
  output += (2 * aligned);
  softmax_c(input, scale, 2, remain, output);
}

/*
 * Assume group is divided by 8
 */
void softmax4_internal(const int8_t* input, float scale,
    unsigned int group, float* output) {
  unsigned int batch = group / 8;

  for (unsigned int i = 0; i < batch; ++i) {
    /* Interleaved load 32 bytes into 4 NEON registers */
    int8x8x4_t q01 = vld4_s8(input);
    /* Convert to 16-bit integers */
    int16x8_t q2 = vmovl_s8(q01.val[0]);
    int16x8_t q3 = vmovl_s8(q01.val[1]);
    int16x8_t q4 = vmovl_s8(q01.val[2]);
    int16x8_t q5 = vmovl_s8(q01.val[3]);
    
    /* Process first 4 groups */
    int16x4_t d10 = vget_low_s16(q2);
    int16x4_t d11 = vget_low_s16(q3);
    int16x4_t d12 = vget_low_s16(q4);
    int16x4_t d13 = vget_low_s16(q5);

    float32x4_t q8 = vcvtq_f32_s32(vmovl_s16(d10));
    float32x4_t q9 = vcvtq_f32_s32(vmovl_s16(d11));
    float32x4_t q10 = vcvtq_f32_s32(vmovl_s16(d12));
    float32x4_t q11 = vcvtq_f32_s32(vmovl_s16(d13));

    q8 = exp_ps(vmulq_n_f32(q8, scale));
    q9 = exp_ps(vmulq_n_f32(q9, scale));
    q10 = exp_ps(vmulq_n_f32(q10, scale));
    q11 = exp_ps(vmulq_n_f32(q11, scale));

    float32x4_t q12 = vaddq_f32(q8, q9);
    q12 = vaddq_f32(q12, q10);
    q12 = vaddq_f32(q12, q11);
    q12 = vrecpeq_f32(q12);

    q8 = vmulq_f32(q12, q8);
    q9 = vmulq_f32(q12, q9);
    q10 = vmulq_f32(q12, q10);
    q11 = vmulq_f32(q12, q11);

    float32x4x4_t b0 = {q8, q9, q10, q11};
    vst4q_f32(output, b0);
    output += 16;

    /* Process last 4 groups */
    d10 = vget_high_s16(q2);
    d11 = vget_high_s16(q3);
    d12 = vget_high_s16(q4);
    d13 = vget_high_s16(q5);

    q8 = vcvtq_f32_s32(vmovl_s16(d10));
    q9 = vcvtq_f32_s32(vmovl_s16(d11));
    q10 = vcvtq_f32_s32(vmovl_s16(d12));
    q11 = vcvtq_f32_s32(vmovl_s16(d13));

    q8 = exp_ps(vmulq_n_f32(q8, scale));
    q9 = exp_ps(vmulq_n_f32(q9, scale));
    q10 = exp_ps(vmulq_n_f32(q10, scale));
    q11 = exp_ps(vmulq_n_f32(q11, scale));

    q12 = vaddq_f32(q8, q9);
    q12 = vaddq_f32(q12, q10);
    q12 = vaddq_f32(q12, q11);
    q12 = vrecpeq_f32(q12);

    q8 = vmulq_f32(q12, q8);
    q9 = vmulq_f32(q12, q9);
    q10 = vmulq_f32(q12, q10);
    q11 = vmulq_f32(q12, q11);

    float32x4x4_t b1 = {q8, q9, q10, q11};
    vst4q_f32(output, b1);
    output += 16;

    input += 32;
  }
}

/*
 * Assume group is divided by 8
 */
void softmax2_internal(const int8_t* input, float scale,
    unsigned int group, float* output) {
  unsigned int batch = group / 8;

  for (unsigned int i = 0; i < batch; ++i) {
    /* Interleaved load 16 bytes into 2 NEON registers */
    int8x8x2_t q0 = vld2_s8(input);
    /* Convert to 16-bit integers */
    int16x8_t q1 = vmovl_s8(q0.val[0]);
    int16x8_t q2 = vmovl_s8(q0.val[1]);
    
    int16x4_t d2 = vget_low_s16(q1);
    int16x4_t d3 = vget_high_s16(q1);
    int16x4_t d4 = vget_low_s16(q2);
    int16x4_t d5 = vget_high_s16(q2);

    /* Process first 4 groups */
    float32x4_t q3 = vcvtq_f32_s32(vmovl_s16(d2));
    float32x4_t q4 = vcvtq_f32_s32(vmovl_s16(d4));
    q3 = exp_ps(vmulq_n_f32(q3, scale));
    q4 = exp_ps(vmulq_n_f32(q4, scale));

    float32x4_t q7 = vaddq_f32(q3, q4);
    q7 = vrecpeq_f32(q7);
    q3 = vmulq_f32(q7, q3);
    q4 = vmulq_f32(q7, q4);

    /* Process last 4 groups */
    float32x4_t q5 = vcvtq_f32_s32(vmovl_s16(d3));
    float32x4_t q6 = vcvtq_f32_s32(vmovl_s16(d5));
    q5 = exp_ps(vmulq_n_f32(q5, scale));
    q6 = exp_ps(vmulq_n_f32(q6, scale));

    float32x4_t q8 = vaddq_f32(q5, q6);
    q8 = vrecpeq_f32(q8);
    q5 = vmulq_f32(q8, q5);
    q6 = vmulq_f32(q8, q6);

    /* Save to memory */
    float32x4x2_t b0 = {q3, q4};
    vst2q_f32(output, b0);
    output += 8;
    float32x4x2_t b1 = {q5, q6};
    vst2q_f32(output, b1);
    output += 8;

    input += 16;
  }
}

/*
 * Assume group is divided by 8
 */
void softmax99_internal(const int8_t* input, float scale,
    unsigned int group, float* output) {
  unsigned int batch = group / 8;

  for (unsigned int i = 0; i < batch; ++i) {
    /* Interleaved load 32 bytes into 4 NEON registers */
    int8x8x4_t q01 = vld4_s8(input);
    /* Convert to 16-bit integers */
    int16x8_t q2 = vmovl_s8(q01.val[0]);
    int16x8_t q3 = vmovl_s8(q01.val[1]);
    int16x8_t q4 = vmovl_s8(q01.val[2]);
    int16x8_t q5 = vmovl_s8(q01.val[3]);
    
    /* Process first 4 groups */
    int16x4_t d10 = vget_low_s16(q2);
    int16x4_t d11 = vget_low_s16(q3);
    int16x4_t d12 = vget_low_s16(q4);
    int16x4_t d13 = vget_low_s16(q5);

    float32x4_t q8 = vcvtq_f32_s32(vmovl_s16(d10));
    float32x4_t q9 = vcvtq_f32_s32(vmovl_s16(d11));
    float32x4_t q10 = vcvtq_f32_s32(vmovl_s16(d12));
    float32x4_t q11 = vcvtq_f32_s32(vmovl_s16(d13));

    q8 = exp_ps(vmulq_n_f32(q8, scale));
    q9 = exp_ps(vmulq_n_f32(q9, scale));
    q10 = exp_ps(vmulq_n_f32(q10, scale));
    q11 = exp_ps(vmulq_n_f32(q11, scale));

    float32x4_t q12 = vaddq_f32(q8, q9);
    q12 = vaddq_f32(q12, q10);
    q12 = vaddq_f32(q12, q11);
    q12 = vrecpeq_f32(q12);

    q8 = vmulq_f32(q12, q8);
    q9 = vmulq_f32(q12, q9);
    q10 = vmulq_f32(q12, q10);
    q11 = vmulq_f32(q12, q11);

    float32x4x4_t b0 = {q8, q9, q10, q11};
    vst4q_f32(output, b0);
    output += 16;

    /* Process last 4 groups */
    d10 = vget_high_s16(q2);
    d11 = vget_high_s16(q3);
    d12 = vget_high_s16(q4);
    d13 = vget_high_s16(q5);

    q8 = vcvtq_f32_s32(vmovl_s16(d10));
    q9 = vcvtq_f32_s32(vmovl_s16(d11));
    q10 = vcvtq_f32_s32(vmovl_s16(d12));
    q11 = vcvtq_f32_s32(vmovl_s16(d13));

    q8 = exp_ps(vmulq_n_f32(q8, scale));
    q9 = exp_ps(vmulq_n_f32(q9, scale));
    q10 = exp_ps(vmulq_n_f32(q10, scale));
    q11 = exp_ps(vmulq_n_f32(q11, scale));

    q12 = vaddq_f32(q8, q9);
    q12 = vaddq_f32(q12, q10);
    q12 = vaddq_f32(q12, q11);
    q12 = vrecpeq_f32(q12);

    q8 = vmulq_f32(q12, q8);
    q9 = vmulq_f32(q12, q9);
    q10 = vmulq_f32(q12, q10);
    q11 = vmulq_f32(q12, q11);

    float32x4x4_t b1 = {q8, q9, q10, q11};
    vst4q_f32(output, b1);
    output += 16;

    input += 32;
  }
}


/*
void softmax4(const int8_t* input, int group, float scale, float* output) {
  int size = group * 4;
  int count = size / 8;
  // int remain = size % 8;
  auto ptr = output;
  // auto p2 = input;
  for (auto i = 0; i < count; ++i) {
    int8x8_t s8v = vld1_s8(input);
    int16x8_t s16v = vmovl_s8(s8v);

    int16x4_t s16v0 = vget_low_s16(s16v);
    int32x4_t s32v0 = vmovl_s16(s16v0);
    float32x4_t f32v0 = vcvtq_f32_s32(s32v0);
    f32v0 = vmulq_n_f32(f32v0, scale);
    v4sf expv0 = exp_ps(f32v0);
    // sum
    float32x2_t sumv0 = vadd_f32(
        vget_high_f32(expv0), vget_low_f32(expv0));
    float32x2_t recv0 = vrecpe_f32(vpadd_f32(sumv0, sumv0));
    expv0 = vmulq_n_f32(expv0, vget_lane_f32(recv0, 0));
    vst1q_f32(ptr, expv0);

    int16x4_t s16v1 = vget_high_s16(s16v);
    int32x4_t s32v1 = vmovl_s16(s16v1);
    float32x4_t f32v1 = vcvtq_f32_s32(s32v1);
    f32v1 = vmulq_n_f32(f32v1, scale);
    v4sf exp_high = exp_ps(f32v1);
    // sum
    float32x2_t sumv1 = vadd_f32(
        vget_high_f32(exp_high), vget_low_f32(exp_high));
    float sum1 = vget_lane_f32(vpadd_f32(sumv1, sumv1), 0);
    exp_high = vmulq_n_f32(exp_high, 1.0/sum1);
    vst1q_f32(ptr+4, exp_high);

    input += 8;
    ptr += 8;
  }
*/

/*
  if (remain > 0) {
    for (auto i = 0; i < remain; ++i) {
      // p1[i] = expf_neon(p2[i]*scale);
      ptr[i] = exp(input[i]*scale);
    }
  }

  ptr = output;
  for (auto i = 0; i < group; ++i) {
    float s = sum4(ptr);
    for (int j = 0; j < 4; ++j) ptr[j] /= s;
    ptr += 4;
  }
*/
//}


/*
void softmax4_safe(const int8_t* input, int group, float scale, float* output) {
  int align8 = (uint64_t)input % 8;
  int align4 = align8 % 4;
  if (align4 % 4) {
    LOG(FATAL) << "Data must be 4-bytes aligned";
  }
  if (align8) {
    softmax_group(input, scale, output);
    input += 4;
    group -= 1;
    output += 4;
  }
  if (group % 2) {
    auto offset = group * 4 - 4;
    softmax_group(input + offset, scale, output + offset);
    group -= 1;
  }
  softmax4(input, group, scale, output);
}
*/

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
v4sf exp_ps(v4sf x) {
  v4sf tmp, fx;

  v4sf one = vdupq_n_f32(1);
  x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
  x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

  /* perform a floorf */
  tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

  /* if greater, substract 1 */
  v4su mask = vcgtq_f32(tmp, fx);
  mask = vandq_u32(mask, vreinterpretq_u32_f32(one));


  fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

  tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
  v4sf z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
  x = vsubq_f32(x, tmp);
  x = vsubq_f32(x, z);

  static const float cephes_exp_p[6] = { c_cephes_exp_p0, c_cephes_exp_p1,
      c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5 };
  v4sf y = vld1q_dup_f32(cephes_exp_p+0);
  v4sf c1 = vld1q_dup_f32(cephes_exp_p+1);
  v4sf c2 = vld1q_dup_f32(cephes_exp_p+2);
  v4sf c3 = vld1q_dup_f32(cephes_exp_p+3);
  v4sf c4 = vld1q_dup_f32(cephes_exp_p+4);
  v4sf c5 = vld1q_dup_f32(cephes_exp_p+5);

  y = vmulq_f32(y, x);
  z = vmulq_f32(x, x);
  y = vaddq_f32(y, c1);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c2);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c3);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c4);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c5);

  y = vmulq_f32(y, z);
  y = vaddq_f32(y, x);
  y = vaddq_f32(y, one);

  /* build 2^n */
  int32x4_t mm;
  mm = vcvtq_s32_f32(fx);
  mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
  mm = vshlq_n_s32(mm, 23);
  v4sf pow2n = vreinterpretq_f32_s32(mm);

  y = vmulq_f32(y, pow2n);
  return y;
}


