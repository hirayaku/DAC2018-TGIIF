#ifndef __MATRIX_H__
#define __MATRIX_H__

/* function description
 * matrix:
 * a00 a01 a02 a03
 * a10 a11 a12 a13
 * a20 a21 a22 a23
 * a30 a31 a32 a33
 * we will get new matrix:
 * tt = nrow*factor
 * b0 = a00+a10+a20+a30
 * b1 = a11+a11+a21+a31
 * b2 = a22+a12+a22+a32
 * b3 = a33+a13+a23+a33
 * b0/tt b1/tt b2/tt b3/tt
 */
void matrix_col_avg1(int nrow, int ncol, const signed char *src, float *dst, int factor);
void matrix_col_avg2(int nrow, int ncol, const signed char *src, float *dst, int factor);
void matrix_col_avg3(int nrow, int ncol, const signed char *src, signed char *dst, int factor);
void matrix_col_avg4(int nrow, int ncol, const signed char *src, signed char *dst, float factor);

void matrix_col_avg1_c(int nrow, int ncol, const signed char *src, float *dst, int factor);
void matrix_col_avg2_c(int nrow, int ncol, const signed char *src, float *dst, int factor);
void matrix_col_avg3_c(int nrow, int ncol, const signed char *src, signed char *dst, int factor);
void matrix_col_avg4_c(int nrow, int ncol, const signed char *src, signed char *dst, float factor);

void matrix_col_avg1_intr(int nrow, int ncol, const signed char *src, float *dst, int factor);
void matrix_col_avg2_intr(int nrow, int ncol, const signed char *src, float *dst, int factor);
void matrix_col_avg3_intr(int nrow, int ncol, const signed char *src, signed char *dst, int factor);
void matrix_col_avg4_intr(int nrow, int ncol, const signed char *src, signed char *dst, float factor);

/*
 * R = A*b
 * A : rows x cols
 * b : cols x 1
 * R : rows x 1
 * store matrix by column order
 * var : variant
 */
void matxvec_rowaccess(int rows, int cols, const float *matA, const float *matB, float *matR);
void matxvec_colaccess(int rows, int cols, const float *matA, const float *matB, float *matR);
void matxvec_add_vec_rowaccess(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void matxvec_add_vec_colaccess(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void matxvec_add_vec_relu_rowaccess(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void matxvec_add_vec_relu_colaccess(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);

void matxvec_rowaccess_c(int rows, int cols, const float *matA, const float *matB, float *matR);
void matxvec_colaccess_c(int rows, int cols, const float *matA, const float *matB, float *matR);
void matxvec_add_vec_rowaccess_c(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void matxvec_add_vec_colaccess_c(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void matxvec_add_vec_relu_rowaccess_c(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void matxvec_add_vec_relu_colaccess_c(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);

void matxvec_rowaccess_intr(int rows, int cols, const float *matA, const float *matB, float *matR);
void matxvec_colaccess_intr(int rows, int cols, const float *matA, const float *matB, float *matR);
void matxvec_add_vec_rowaccess_intr(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void matxvec_add_vec_colaccess_intr(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void matxvec_add_vec_relu_rowaccess_intr(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void matxvec_add_vec_relu_colaccess_intr(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);

/*
 * R = a*B + c
 * a : 1 x rows
 * B : rows x cols
 * c : 1 x cols
 * R : 1 x cols
 */
void vecxmat_add_vec_rowaccess(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void vecxmat_add_vec_colaccess(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void vecxmat_add_vec_relu_rowaccess(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void vecxmat_add_vec_relu_colaccess(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);

void vecxmat_add_vec_rowaccess_c(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void vecxmat_add_vec_colaccess_c(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void vecxmat_add_vec_relu_rowaccess_c(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void vecxmat_add_vec_relu_colaccess_c(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);

void vecxmat_add_vec_rowaccess_intr(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void vecxmat_add_vec_colaccess_intr(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void vecxmat_add_vec_relu_rowaccess_intr(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);
void vecxmat_add_vec_relu_colaccess_intr(int rows, int cols, const float *matA, const float *matB, const float *matC, float *matR);

#endif /*__MATRIX_H__ */
