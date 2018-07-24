# Prerequisites

Please follow the steps below to modify the boot setting and install the dnndk package on PYNQ-Z1. It has been tested on PYNQ-Z1 Image v2.1. 

- Copy and change the `devicetree.dtb` in the SD\_CARD.

- Stop the autostart at the booting stage, and then type the cmds below:
```sh     
setenv bootargs 'console=ttyPS0,115200 mem=256M root=/dev/mmcblk0p2 rw earlyprintk rootfstype=ext4 rootwait devtmpfs.mount=1 uio_pdrv_genirq.of_id="generic-uio"'
setenv fdt_high 0x10000000
setenv initrd_high 0x10000000
saveenv
pri
```
and then type cmd: 
```sh
reset
```
system should boot successfully, After the linux get started, use cmd
```sh
free -h
```
make sure that system available memory is 256M instead of 1024M.

- Install the dnndk package.
copy and unzip the dnndk-lib.zip to the board. 
use cmd: sudo ./install.sh to install the dnndk package. Make sure that you see the successful info without any warning or error.

- Run our code in ipynb.
