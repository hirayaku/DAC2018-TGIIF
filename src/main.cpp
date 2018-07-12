/*
 * Copyright (c) 2016-2017 DeePhi Tech, Inc.
 *
 * All Rights Reserved. No part of this source code may be reproduced
 * or transmitted in any form or by any means without the prior written
 * permission of DeePhi Tech, Inc.
 *
 * Filename: main.cc
 * Version: 1.07 beta
 * Description:
 * Sample source code showing how to deploy SSD neural network on
 * DeePhi DPU@Zynq7020 platform.
 */

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>

// Header file OpenCV for image processing
#include <opencv2/opencv.hpp>
// Header files for DNNDK APIs
#include <dnndk/dputils.h>
#include <dnndk/n2cube.h>
#include <dnndk/transform.h>
#include "ssd_detector.hpp"
#include "prior_boxes.hpp"
#include "neon_math.hpp"
#include "exp_lut.hpp"
#include "SoftmaxTable.hpp"

using namespace std;
using namespace cv;
using namespace deephi;

// DPU Kernel name for SSD Convolution layers
#define KERNEL_CONV "ssd"
// DPU node name for input and output
#define CONV_INPUT_NODE "conv1_1"
#define CONV_OUTPUT_NODE_LOC "mbox_loc"
#define CONV_OUTPUT_NODE_CONF "mbox_conf"

// detection params
const float NMS_THRESHOLD = 0.45;
const float CONF_THRESHOLD = 0.01;
const int TOP_K = 1;
const int KEEP_TOP_K = 1;
const int num_classes = 99;

extern "C" {
typedef struct {
    int label;
    int xmin;
    int xmax;
    int ymin;
    int ymax;
    float confidence;
} result_t;
}

typedef struct {
    int8_t score;
    int row;
    int col;
} max_t;

/**
 * @Optimized max-find for **int8_t** arrays using data coalescing
 * 
 * @param array - the 2D int8_t array
 * @param rows - the rows of the array
 * @param cols - the cols of the array, must be divisable by 4
 */
max_t find_max_int8(int8_t* array, int rows, int cols) {
  int row = -1, col = -1;
  int8_t max_value = 0;
  unsigned *packed_array = reinterpret_cast<unsigned *>(array);
  for(int i = 0; i < rows * cols / 4; i++) {
    unsigned data = packed_array[i];
    int8_t data0 = data & 0x000000ff;
    int8_t data1 = (data & 0x0000ff00) >> 8;
    int8_t data2 = (data & 0x00ff0000) >> 16; 
    int8_t data3 = (data & 0xff000000) >> 24;
    if(data0 > max_value) {
        max_value = data0;
        row = i * 4;
    }
    if(data1 > max_value) {
        max_value = data1;
        row = i * 4 + 1;
    }
    if(data2 > max_value) {
        max_value = data2;
        row = i * 4 + 2;
    }
    if(data3 > max_value) {
        max_value = data3;
        row = i * 4 + 3;
    }
  }
  // In case the array is not 4-byte aligned
  for(int i =  rows * cols / 4 * 4; i < rows * cols; i++) {
      int8_t data = array[i];
      if(data > max_value) {
          max_value  = data;
          row = i;
      }
  }
  col = row - (row / cols * cols);
  row = row / cols;
  return {max_value, row, col};
}

/**
 * @brief Calculate softmax on CPU
 *
 * @param src - pointer to int8_t DPU data to be calculated
 * @param size - size of input int8_t DPU data
 * @param scale - scale to miltiply to transform DPU data from int8_t to float
 * @param dst - pointer to float result after softmax
 *
 * @return none
 */
void CPUSoftmax(int8_t* src, int size, float scale, float* dst) {
  float sum = 0.0f;
  for (auto i = 0; i < size; ++i) {
    dst[i] = exp(src[i] * scale);
    sum += dst[i];
  }
  for (auto i = 0; i < size; ++i) {
    dst[i] /= sum;
  }
}

void CreatePriors(vector<shared_ptr<vector<float>>> *priors) {
  vector<float> variances{0.1, 0.1, 0.2, 0.2};
  vector<PriorBoxes> prior_boxes;

  // prior boxes for model: 90% compress rate, 0.7 scale img input
  prior_boxes.emplace_back(PriorBoxes{
      448, 252, 56, 32, variances, {30}, {66}, {2}, 0.5, 8.0, 8.0});

  prior_boxes.emplace_back(PriorBoxes{
      448, 252, 28, 16, variances, {66}, {127}, {2, 3}, 0.5, 16.0, 16.0});

  prior_boxes.emplace_back(PriorBoxes{
      448, 252, 14, 8, variances, {127}, {188}, {2, 3}, 0.5, 32.0, 32.0});

  prior_boxes.emplace_back(PriorBoxes{
      448, 252, 7, 4, variances, {188}, {249}, {2, 3}, 0.5, 64.0, 64.0});

  int num_priors = 0;
  for (auto &p : prior_boxes) {
    num_priors += p.priors().size();
  }

  priors->clear();
  priors->reserve(num_priors);
  for (auto i = 0U; i < prior_boxes.size(); ++i) {
    priors->insert(priors->end(), prior_boxes[i].priors().begin(),
                   prior_boxes[i].priors().end());
  }
}

class DPU_Handler {
    DPUKernel *kernel_conv;
    DPUTask *task_conv;

    vector<shared_ptr<vector<float>>> priors;
    int8_t *conf_mem = nullptr;
    float *conf_softmax = nullptr;
    int size;   // output tensor size
    float mean[3] = {104, 117, 123};

    SoftmaxTable *STable;

    // Sync variables for overlapping cv.imread and RunSSD
    mutex load_mutex, run_mutex;
    condition_variable load_cv, run_cv;
    bool load_to_begin, load_finished;
    Mat *even_img, *odd_img;

    // Output bbox results
    vector<result_t> results;

    // DDR_0 --> DDR_dpu
    // Not using dpuSetInputImage in dputils.cpp, since it wastes us another 5ms per call
    void set_input(Mat &img) {
        // Set image into CONV Task with mean value
        auto time0 = chrono::system_clock::now();

        int8_t *input_addr = dpuGetInputTensorAddress(task_conv, CONV_INPUT_NODE);
        float scale_fix = dpuGetInputTensorScale(task_conv, CONV_INPUT_NODE);
        transform_bgr(img.cols, img.rows, img.data,
                      input_addr,
                      mean[0], scale_fix,
                      mean[1], scale_fix,
                      mean[2], scale_fix
                     );
        //dpuSetInputImage(task_conv, (char*)CONV_INPUT_NODE, img, mean);
        auto time1 = chrono::system_clock::now();
        cout << "setinput: " << chrono::duration_cast<chrono::microseconds>(time1-time0).count() << ".us" << endl;
    }
    // Signal DPU to run task: DDR_dpu --> BRAM --> DDR_dpu'
    inline void run_dpu_task() {
        auto time0 = chrono::system_clock::now();
        dpuRunTask(this->task_conv);
        auto time1 = chrono::system_clock::now();
        cout << "DPU     : " << chrono::duration_cast<chrono::microseconds>(time1-time0).count() << ".us" << endl;
    }

    result_t post_process(const Mat& img) {
        // Initializations
        result_t top = {0, 0, 0, 0, 0, 0};
        int8_t* loc =
            (int8_t*)dpuGetOutputTensorAddress(task_conv, CONV_OUTPUT_NODE_LOC);
        int8_t* conf =
            (int8_t*)dpuGetOutputTensorAddress(task_conv, CONV_OUTPUT_NODE_CONF);
        float loc_scale = dpuGetOutputTensorScale(task_conv, CONV_OUTPUT_NODE_LOC);
        float conf_scale =
          dpuGetOutputTensorScale(task_conv, CONV_OUTPUT_NODE_CONF);

        auto time2 = chrono::system_clock::now();

        int count = size/num_classes;
        memcpy(conf_mem, conf, this->size);
        auto t = STable->cal_softmax(conf_mem);
        int location = get<0>(t);
        int8_t *max_row = conf_mem + location * num_classes;
        auto max_pt = max_element(max_row + 1, max_row + num_classes);
        auto dis = distance(max_row, max_pt);
        int classification = dis;

        /*
        auto max_data = find_max_int8(conf_mem, count, num_classes);
        int location = max_data.row;
        int classification = max_data.col;
        */

        auto time3 = chrono::system_clock::now();
        CPUSoftmax(conf_mem + location*num_classes, num_classes, conf_scale, conf_softmax + location*num_classes);
        
        auto time4 = chrono::system_clock::now();
        MultiDetObjects results;
        vector<float> th_conf(num_classes, CONF_THRESHOLD);
        SSDdetector* detector_ = new SSDdetector(num_classes,SSDdetector::CodeType::CENTER_SIZE, false,
                  KEEP_TOP_K, th_conf, TOP_K, NMS_THRESHOLD, 1.0, priors, loc_scale);
        detector_->Detect(loc, conf_softmax, &results, location, classification);

        float top_conf = 0;
        float full_cols = img.cols * 10 / 7;
        float full_rows = img.rows * 10 / 7;
        for (size_t i = 0; i < results.size(); ++i) {
            int label = get<0>(results[i]);
            float xmin = get<2>(results[i]).x * full_cols;
            float ymin = get<2>(results[i]).y * full_rows;
            float xmax = xmin + (get<2>(results[i]).width) * full_cols;
            float ymax = ymin + (get<2>(results[i]).height) * full_rows;
            xmin = round(xmin*100.0)/100.0;
            ymin = round(ymin*100.0)/100.0;
            xmax = round(xmax*100.0)/100.0;
            ymax = round(ymax*100.0)/100.0;
            float confidence = get<1>(results[i]);

            xmin = std::min(std::max(xmin, 0.0f), full_cols);
            xmax = std::min(std::max(xmax, 0.0f), full_cols);
            ymin = std::min(std::max(ymin, 0.0f), full_rows);
            ymax = std::min(std::max(ymax, 0.0f), full_rows);

            if (top_conf < confidence) {
                top_conf = confidence;
                top.label = label;
                top.xmin = (int)xmin; top.xmax = (int)xmax;
                top.ymin = (int)ymin; top.ymax = (int)ymax;
                top.confidence = confidence;
            }
        }

        auto time5 = chrono::system_clock::now();
        //cout << "before  : " << chrono::duration_cast<chrono::microseconds>(time0-timex).count() << ".us" << endl;
        //cout << "dpu time: " << chrono::duration_cast<chrono::microseconds>(time2-time1).count() << ".us" << endl;
        cout << "find    : " << chrono::duration_cast<chrono::microseconds>(time3-time2).count() << ".us" << endl;
        cout << "softmax : " << chrono::duration_cast<chrono::microseconds>(time4-time3).count() << ".us" << endl;
        cout << "detect  : " << chrono::duration_cast<chrono::microseconds>(time5-time4).count() << ".us" << endl;
        cout << "Post    : " << chrono::duration_cast<chrono::microseconds>(time5-time2).count() << ".us" << endl;
        return top;
    }

    // thread-1: run dpu task when t2 is ready
    void t1_run_task(unsigned total_count) {
        Mat *current_img;
        for(unsigned current = 0; current < total_count; current++) {
            if(current%2 == 0) {
                current_img = this->even_img;
            } else
                current_img = this->odd_img;

            {
                unique_lock<mutex> lk(this->run_mutex);
                while(!load_finished)
                    this->run_cv.wait(lk);
                load_finished = false;
            }

#ifdef DEBUG
            cout << "Input    : " << to_string(current) << ".jpg" << endl;
#endif

            // DDR_0 --> DDR_dpu when DDR_0 is ready
            this->set_input(*current_img);
            // Then make t2 to process DDR_dpu' and load next img
            {
                lock_guard<mutex> guard(this->load_mutex);
                load_to_begin = true;
            }
            this->load_cv.notify_one();
            this->run_dpu_task();

#ifdef DEBUG
            cout << "- - - - -" << endl;
#endif
        }
        {
            unique_lock<mutex> lk(this->run_mutex);
            while(!load_finished)
                this->run_cv.wait(lk);
            load_finished = false;
        }
        {
            lock_guard<mutex> guard(this->load_mutex);
            load_to_begin = true;
        }
        this->load_cv.notify_one();
    }
    // thread-2: load img
    void t2_post_and_load(string imgs_dir, vector<string> &imgs_vec, unsigned total_count) {
        Mat *current_img, *last_img;
        for(unsigned current = 0; current < total_count; current++) {
            if(current%2 == 0) {
                current_img = this->even_img;
                last_img = this->odd_img;
            } else {
                current_img = this->odd_img;
                last_img = this->even_img;
            }

            // Start post processing last DDR_dpu'
            if(current > 1) {
                this->results.push_back(this->post_process(*last_img));
            }
            Mat read_img = imread(imgs_dir + "/" + imgs_vec[current]);
            resize(read_img, *current_img, Size(), 0.7, 0.7);

            {
                lock_guard<mutex> guard(this->run_mutex);
                load_finished = true;
            }
            this->run_cv.notify_one();
            // Wait until some data is ready
            {
                unique_lock<mutex> lk(this->load_mutex);
                while(!load_to_begin)
                    this->load_cv.wait(lk);
                load_to_begin = false;
            }
        }
        this->results.push_back(this->post_process(*this->odd_img));
        {
            lock_guard<mutex> guard(this->run_mutex);
            load_finished = true;
        }
        this->run_cv.notify_one();
        // Wait until some data is ready
        {
            unique_lock<mutex> lk(this->load_mutex);
            while(!load_to_begin)
                this->load_cv.wait(lk);
            load_to_begin = false;
        }
        this->results.push_back(this->post_process(*this->even_img));
    }

public:
  DPU_Handler(string lib_path) {
    dpuOpen();
    this->kernel_conv = dpuLoadKernelModel(KERNEL_CONV, lib_path.data()); //"/home/xilinx/ssd_99_py/model/dpu_ssd_233.elf"
    this->task_conv = dpuCreateTask(this->kernel_conv, 0);
    CreatePriors(&(this->priors));
    this->size = dpuGetOutputTensorSize(task_conv, CONV_OUTPUT_NODE_CONF);
    this->conf_mem = new int8_t[size];
    this->conf_softmax = new float[size];

    //float conf_scale =
    //    dpuGetOutputTensorScale(task_conv, CONV_OUTPUT_NODE_CONF);
    this->STable = new SoftmaxTable(0.125, 1058904, 99);
    //this->STable = new SoftmaxTable(, this->size, num_classes);

    this->load_to_begin = false;
    this->load_finished = false;
    this->even_img = new Mat;
    this->odd_img = new Mat;

    cout << "DPU InputSize : " << dpuGetInputTensorSize(task_conv, (char*)CONV_INPUT_NODE) << endl;
    cout << "DPU OutputSize: " << this->size << endl;
    cout << "- - - - - - - - - -" << endl;
  }

  ~DPU_Handler() {
    delete STable;
    delete odd_img;
    delete even_img;
    delete [] this->conf_softmax;
    delete [] this->conf_mem;
    // Destroy DPU Tasks and Kernels and free resources
    dpuDestroyTask(this->task_conv);
    dpuDestroyKernel(this->kernel_conv);
    // Detach from DPU driver and release resources
    dpuClose();
  }
  void dpu_clear() {
    this->results.clear();
    this->load_to_begin = false;
    this->load_finished = false;
  }
  result_t dpu_detect_single(string img_path) {
    Mat img = imread(img_path.data());
    set_input(img);
    run_dpu_task();
    auto r = post_process(img);
    results.push_back(r);
    return r;
  }
  void dpu_detect_list(string imgs_dir, vector<string> &imgs_vec, unsigned img_count) {
    dpu_clear();
    // Spawn threads
    thread t1(&DPU_Handler::t1_run_task, this, img_count);
    thread t2(&DPU_Handler::t2_post_and_load, this, imgs_dir, ref(imgs_vec), img_count);

    // set cpu affinity
    cpu_set_t cpuset;
    int rc;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    rc = pthread_setaffinity_np(t1.native_handle(),
                                sizeof(cpuset),
                                &cpuset);
    if(rc != 0) {
        cerr << "Fail to bind t1_run_task to CPU0!" << endl;
    }

    CPU_SET(1, &cpuset);
    rc = pthread_setaffinity_np(t2.native_handle(),
                                sizeof(cpuset),
                                &cpuset);
    if(rc != 0) {
        cerr << "Fail to bind t2_post_and_load to CPU1!" << endl;
    }
    // set thread priority
    struct sched_param sp;
    sp.sched_priority = 2;
    pthread_setschedparam(t1.native_handle(), SCHED_FIFO, &sp);
    sp.sched_priority = 2;
    pthread_setschedparam(t2.native_handle(), SCHED_FIFO, &sp);

    // Wait for threads to finish
    t2.join();
    t1.join();
  }
  result_t *dpu_get_results() {
      return this->results.data();
  }
};

DPU_Handler *dpu_ptr = nullptr;
extern "C" {
  void dpu_initialize(char *c_lib_path) {
    dpu_ptr = new DPU_Handler(string(c_lib_path));
  }
  void dpu_destroy() {
    dpu_ptr->~DPU_Handler();
    dpu_ptr = nullptr;
  }
  void dpu_clear() {
      dpu_ptr->dpu_clear();
  }
  result_t dpu_detect_single(char *c_img_path) {
    auto time_begin = chrono::system_clock::now();
    assert(dpu_ptr != nullptr);
    auto r = dpu_ptr->dpu_detect_single(string(c_img_path));
    auto time_end = chrono::system_clock::now();
    cout << "Overall: " << chrono::duration_cast<chrono::microseconds>(time_end - time_begin).count() << ".us" << endl;
    cout << "- - - - -" << endl;
    return r;
  }
  void dpu_detect_list(char *c_imgs_file, unsigned num) {
      auto time_begin = chrono::system_clock::now();

      //auto time0 = chrono::system_clock::now();
      string imgs_dir, temp;
      vector<string> imgs;
      ifstream imgs_file(c_imgs_file);
      // Get imgs directory
      getline(imgs_file, imgs_dir);
      while(getline(imgs_file, temp))
          imgs.push_back(temp);
      //auto time1 = chrono::system_clock::now();

      dpu_ptr->dpu_detect_list(imgs_dir, imgs, num);

      auto time_end = chrono::system_clock::now();
      //auto read_micro_sec = chrono::duration_cast<chrono::microseconds>(time1 - time0).count();
      auto micro_sec = chrono::duration_cast<chrono::microseconds>(time_end - time_begin).count();
      cout << "- - - - - - - - - - " << endl;
      cout << "Overall : " << micro_sec << ".us" << endl;
      cout << "FPS     : " << to_string(num / (micro_sec / 1000000.0f)) << endl;
  }
  result_t* dpu_get_results() {
      return dpu_ptr->dpu_get_results();
  }
}
