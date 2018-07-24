# DAC 2018 System Design Contest-TGIIF
The 1st place winner's source codes for DAC 2018 System Design Contest, FPGA Track. Our design is based on DeePhi DPU RTL IP Core and DeePhi DNNDK software stack. For more infos about DeePhi DPU and DNNDK, please refer to [www.deephi.com][1].

- For prerequisites, refer to `prerequisites` folder. They are necessary for running our demo on PYNQ-Z1.

- For the tutorial in python, refer to the python notebook in `TGIIF` folder.

- For the source codes of software, our block design and the NN model we used, refer to `src`, `block_design`, and `model` folders, respectively.

## Algorithm
We used SSD(Single Shot Multibox Detector) as our based algorithm, and modified it to better fit for the acceleration on DPU. The overview of the modification applied to the SSD network is showed below.

![ssd](https://github.com/hirayaku/DAC2018-TGIIF/raw/master/image/ssd.png)


[1]:http://www.deephi.com/  "www.deephi.com"
