#ifndef __TIME_HELPER_HPP__
#define __TIME_HELPER_HPP__

#include <sys/time.h>

#define __AVE_TIC__(tag) static int ____##tag##_total_time=0; \
        static int ____##tag##_total_conut=0;\
        timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __AVE_TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        ____##tag##_total_conut++; \
        ____##tag##_total_time+=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        fprintf(stderr,  #tag ": %d us\n", ____##tag##_total_time/____##tag##_total_conut);

#define __TIC__(tag) timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        int ____##tag##_total_time=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        fprintf(stderr,  #tag ": %d us\n", ____##tag##_total_time);

#endif // __TIME_HELPER_HPP__