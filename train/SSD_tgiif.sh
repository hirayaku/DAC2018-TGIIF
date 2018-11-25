#!bin/sh

PATH_TO_CAFFE=''

$PATH_TO_CAFFE/build/tools/caffe train \
    --solver="solver.prototxt" \
    --weights="VGG16_BN_FIXED_PRETRAINED.caffemodel" \
    --gpu 0,1,2,3 2>&1 | tee logs/log_SSD_tgiif.log
