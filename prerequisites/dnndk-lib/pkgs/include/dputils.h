/* 
 * Copyright (c) 2016-2018 DeePhi Tech, Inc.
 *
 * All Rights Reserved. No part of this source code may be reproduced
 * or transmitted in any form or by any means without the prior written
 * permission of DeePhi Tech, Inc.
 *
 * Filename: dputils.h
 * Version: 1.10 beta
 *
 * Description:
 * Header file containing all the exported APIs of DNNDK utility library libdputils
 * Please refer to document "deephi_dnndk_user_guide.pdf" for more details of APIs.
 */

#ifndef _DPUTILS_H_
#define _DPUTILS_H_

#include <opencv2/opencv.hpp>

struct  dpu_task_t;
typedef struct dpu_task_t   DPUTask;


/* Set image into DPU Task's input Tensor */
int dpuSetInputImage(DPUTask *task, const char *nodeName,
    const cv::Mat &image, float *mean);

/* Set image into DPU Task's input Tensor with a specified scale parameter */
int dpuSetInputImageWithScale(DPUTask *task, const char *nodeName,
    const cv::Mat &image, float *mean, float scale);

/* Set image into DPU Task's input Tensor (mean values automatically processed by N2Cube) */
int dpuSetInputImage2(DPUTask *task, const char *nodeName, const cv::Mat &image);

#endif
