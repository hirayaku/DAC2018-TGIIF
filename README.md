# DAC 2018 System Design Contest-TGIIF
The 1st place winner's source codes for DAC 2018 System Design Contest, FPGA Track. Our design is based on DeePhi DPU RTL IP Core and DeePhi DNNDK software stack. For more infos about DeePhi DPU and DNNDK, please refer to [www.deephi.com][1].

- For prerequisites, refer to `prerequisites` folder. They are necessary for running our demo on PYNQ-Z1.

- For the tutorial in python, refer to the python notebook in `TGIIF` folder.

- For the source codes of software, our block design and the NN model we used, refer to `src`, `block_design`, and `model` folders, respectively.

## Algorithm
### SSD Modification
We used SSD(Single Shot Multibox Detector) as our based algorithm, and modified it to better fit for the acceleration on DPU. The overview of the modification applied to the SSD network is showed below.

![ssd](https://github.com/hirayaku/DAC2018-TGIIF/raw/master/image/ssd.png)

There are mainly four modifications as below, and you can find more details in the `model/caffe_model` folder.

- Better performance-->Resize input image from 640x360 to 448x252 (factor=0.7).

- Small objects-->Delete deep layer branches to speed up and get higher IOU.

- Speed up the convergence-->Add batch normalization

- Evaluation metric-->From mAP to IOU

### Pruning and Quantization
We used DeePhi DNNDK, the first public release of deep learning SDK in China, to support the full-stack development and deployment on DeePhi DPU platform. More infos about DeePhi DNNDK, refer to [www.deephi.com/dnndk.html][2].

![dnndk](https://github.com/hirayaku/DAC2018-TGIIF/raw/master/image/dnndk.png)

In the phase of training, we use the fixed-point instead of the float-point for data representation as showed below. The processes of pruning and quantization are implemented via DeePhi DNNDK.

![training](https://github.com/hirayaku/DAC2018-TGIIF/raw/master/image/training.png)

## Software
We used DeePhi DNNDK for NN model compilation and runtime deployment. Furthermore, we applied several optimization methods to improve the accuracy and speed on PYNQ-Z1.
### Funtional Level Optimization
- Max-value Selection: Selecting the max confidence group bounding box instead of compute all the NMS.

- Table look-up: Computing softmax with table look-up, reduce the exponent arithmetic time.

### System Level Optimization
Divide the workflow into fine-grained sub-tasks & re-organization using multi-threading (1.8x speed up with 2 threads)

![multithread](https://github.com/hirayaku/DAC2018-TGIIF/raw/master/image/multithread.png)

## Hardware
### Overview of DPU
We use DeePhi DPU IP as our hardware system, it is the basis of our design. It is a Deep-Learning Processor Unit that is specially designed for CNN and DNN, named as Aristotle. 

![dpu](https://github.com/hirayaku/DAC2018-TGIIF/raw/master/image/dpu.png)

### Block Design
Here is our hardware solution on PYNQ-Z1. As for the connectivity there are two 64bit ports used for DDR read/write and one 32bit port for instruction and profiler. ARM CPU use one 32bit port to read/write register for controling. 

![bd](https://github.com/hirayaku/DAC2018-TGIIF/raw/master/image/bd.png)

[1]:http://www.deephi.com/  "deephi"
[2]:http://www.deephi.com/dnndk.html  "dnndk"
