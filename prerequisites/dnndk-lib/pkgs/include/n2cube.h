 /* 
  * Copyright (c) 2016-2018 DeePhi Tech, Inc.
  *
  * All Rights Reserved. No part of this source code may be reproduced
  * or transmitted in any form or by any means without the prior written
  * permission of DeePhi Tech, Inc.
  *
  * Filename: n2cube.h
  * Version: 1.10 beta
  *
  * Description:
  * Header file containing all the exported APIs of DNNDK Runtime library libn2cube
  * Please refer to document "deephi_dnndk_user_guide.pdf" for more details of these APIs.
  */
#ifndef _N2CUBE_H_
#define _N2CUBE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>


/* DPU Task runtime mode definitions */

/* Task in normal mode (defaul mode) */
#define T_MODE_NORMAL        (0)

/* Task in profiling mode in order to collect performance stastics for each DPU Node */
#define T_MODE_PROFILE       (1<<0)  

/* Task in debug mode in order to dump each Node's Code/Bias/Weights/Input/Output raw data for debugging */
#define T_MODE_DEBUG         (1<<1)
                                        
/* Exported data structures of DPU Kernel/Task/Tensor */
struct  dpu_kernel_t;
struct  dpu_task_t;
struct  task_tensor_t;

typedef struct dpu_kernel_t  DPUKernel;
typedef struct dpu_task_t    DPUTask;
typedef struct task_tensor_t DPUTensor;


/* Open & initialize the usage of DPU device */
int dpuOpen();

/* Close & finalize the usage of DPU device */
int dpuClose();

/* Load a DPU Kernel and allocate DPU memory space for 
   its Code/Weight/Bias segments */
DPUKernel *dpuLoadKernel(const char *netName);
DPUKernel *dpuLoadKernelModel(const char *netName,const char* modelPath);

/* Set mean values for DPU Kernel */
int dpuSetKernelMeanValue(DPUKernel *kernel, int mean1, int mean2, int mean3);

/* Destroy a DPU Kernel and release its associated resources */
int dpuDestroyKernel(DPUKernel *kernel);

/* Instantiate a DPU Task from one DPU Kernel, allocate its private
   working memory buffer and prepare for its execution context */
DPUTask *dpuCreateTask(DPUKernel *kernel, int mode);

/* Launch the running of DPU Task */
int dpuRunTask(DPUTask *task);

/* Remove a DPU Task, release its working memory buffer and destroy
   associated execution context */
int dpuDestroyTask(DPUTask *task);

/* Enable dump facility of DPU Task while running for debugging purpose */
int dpuEnableTaskDebug(DPUTask *task);

/* Enable profiling facility of DPU Task while running to get its performance metrics */
int dpuEnableTaskProfile(DPUTask *task);

/* Get the execution time of DPU Task */
long long dpuGetTaskProfile(DPUTask *task);

/* Get the execution time of DPU Node */
long long dpuGetNodeProfile(DPUTask *task, const char*nodeName);

/* Get input Tensor of DPU Task */
DPUTensor* dpuGetInputTensor(DPUTask *task, const char*nodeName);

/* Get the start address of DPU Task's input Tensor */
int8_t* dpuGetInputTensorAddress(DPUTask *task, const char *nodeName);

/* Get the size (in byte) of one DPU Task's input Tensor */
int dpuGetInputTensorSize(DPUTask *task, const char *nodeName);

/* Get the scale value (DPU INT8 quantization) of one DPU Task's input Tensor */
float dpuGetInputTensorScale(DPUTask *task, const char *nodeName);

/* Get the height dimension of one DPU Task's input Tensor */
int dpuGetInputTensorHeight(DPUTask *task, const char *nodeName);

/* Get the width dimension of one DPU Task's input Tensor */
int dpuGetInputTensorWidth(DPUTask *task, const char *nodeName);

/* Get the channel dimension of one DPU Task's input Tensor */
int dpuGetInputTensorChannel(DPUTask *task, const char *nodeName);

/* Get output Tensor of one DPU Task */
DPUTensor* dpuGetOutputTensor(DPUTask *task, const char *nodeName);

/* Get the start address of one DPU Task's output Tensor */
int8_t* dpuGetOutputTensorAddress(DPUTask *task, const char *nodeName);

/* Get the size (in byte) of one DPU Task's output Tensor */
int dpuGetOutputTensorSize(DPUTask *task, const char *nodeName);

/* Get the scale value (DPU INT8 quantization) of one DPU Task's output Tensor */
float dpuGetOutputTensorScale(DPUTask *task, const char *nodeName);

/* Get the height dimension of one DPU Task's output Tensor */
int dpuGetOutputTensorHeight(DPUTask *task, const char *nodeName);

/*  Get the channel dimension of one DPU Task's output Tensor */
int dpuGetOutputTensorWidth(DPUTask *task, const char *nodeName);

/* Get DPU Node's output tensor's channel */
int dpuGetOutputTensorChannel(DPUTask *task, const char *nodeName);

/* Get the size of one DPU Tensor */
int dpuGetTensorSize(DPUTensor* tensor);

/* Get the start address of one DPU Tensor */
int8_t* dpuGetTensorAddress(DPUTensor* tensor);

/* Get the scale value of one DPU Tensor */
float dpuGetTensorScale(DPUTensor* tensor);

/* Get the height dimension of one DPU Tensor */
int dpuGetTensorHeight(DPUTensor* tensor);

/* Get the width dimension of one DPU Tensor */
int dpuGetTensorWidth(DPUTensor* tensor);

/* Get the channel dimension of one DPU Tensor */
int dpuGetTensorChannel(DPUTensor* tensor);

/* Set DPU Task's input Tensor with data stored under Caffe
   Blob's order (channel/height/width) in INT8 format */
int dpuSetInputTensorInCHWInt8(DPUTask *task, const char *nodeName, int8_t *data, int size);

/* Set DPU Task's input Tensor with data stored under Caffe
   Blob's order (channel/height/width) in FP32 format */
int dpuSetInputTensorInCHWFP32(DPUTask *task, const char *nodeName, float *data, int size);

/* Set DPU Task's input Tensor with data stored under DPU
   Tensor's order (height/width/channel) in INT8 format */
int dpuSetInputTensorInHWCInt8(DPUTask *task, const char *nodeName, int8_t *data, int size);

/* Set DPU Task's input Tensor with data stored under DPU
   Tensor's order (height/width/channel) in FP32 format */
int dpuSetInputTensorInHWCFP32(DPUTask *task, const char *nodeName, float *data, int size);

/* Get DPU Task's output Tensor and store them under Caffe
   Blob's order (channel/height/width) in INT8 format */
int dpuGetOutputTensorInCHWInt8(DPUTask *task, const char *nodeName, int8_t *data, int size);

/* Get DPU Task's output Tensor and store them under Caffe
   Blob's order (channel/height/width) in FP32 format */
int dpuGetOutputTensorInCHWFP32(DPUTask *task, const char *nodeName, float *data, int size);

/* Get DPU Task's output Tensor and store them under DPU
   Tensor's order (height/width/channel) in INT8 format */
DPUTensor* dpuGetOutputTensorInHWCInt8(DPUTask *task, const char *nodeName);

/* Get DPU Task's output Tensor and store them under DPU
   Tensor's order (height/width/channel) in FP32 format */
int dpuGetOutputTensorInHWCFP32(DPUTask *task, const char *nodeName, float *data, int size);

/* DEPRECATED, use with caution! */
int dpuRunSoftmax( DPUTask *task, const char *nodeName, float* softmax);

#ifdef __cplusplus
}
#endif

#endif
