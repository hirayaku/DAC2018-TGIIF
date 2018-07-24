#!/bin/bash
echo "Begin to install DeePhi DNNDK ..."

arch=$(uname -m)
if [ "$arch" != "armv7l" ]; then
    echo "DeePhi DNNDK package could only be installed on ARM targes."
    echo "Please contact dnndk-support@deephi.com for more help."
    echo "Terminate installation ..."
    exit
fi

###########################################################
echo "Install DeePhi DPU Driver ..."

spt_ver_ar=(3.17.0,4.9.0)   # support version list
sysver_pref=$(uname -r | awk -F'[.-]' 'BEGIN {}; {print $1 "." $2 "." $3}')

mkdir -p /lib/modules/$(uname -r)/extra/
touch /lib/modules/$(uname -r)/modules.order
touch /lib/modules/$(uname -r)/modules.builtin

if [[ "${spt_ver_ar[@]}" =~ $sysver_pref ]] ; then
    cp pkgs/driver/dpu-$sysver_pref.ko /lib/modules/$(uname -r)/extra/dpu.ko
else
    echo "Linux kernel version "  $(uname -r) "not support!!"
	exit
fi

depmod -a
rst="$(lsmod | grep dpu 2>&1)"
if [ -n "$rst" ] ; then
	rmmod dpu
fi 
rst="$(modprobe dpu | grep modprobe 2>&1)"
if [ -n "$rst" ] ; then
	echo $rst
	exit
fi 

if ! grep -Fxq "dpu" /etc/modules ; then
    sh -c 'echo "dpu" >> /etc/modules' ;
fi

###########################################################
echo "Install DeePhi tools, runtime & libraries ..."
cp pkgs/bin/*  /usr/local/bin/
cp pkgs/lib/*  /usr/local/lib/

lfile="/usr/local/lib/libdputils.so"
if [ -f $lfile ] ; then
    rm $lfile
fi
ln -s /usr/local/lib/libdputils.so.3.1 $lfile 
mkdir -p /usr/local/include/dnndk/
cp pkgs/include/*.h  /usr/local/include/dnndk/ 
ldconfig

echo "Complete installation successfully."
