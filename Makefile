PROJECT   =    libssd.so
OBJ       :=   main.o  dpu_ssd.elf ssd_detector.o prior_boxes.o

CXX       :=   g++
CC        :=   gcc

# linking libraries of OpenCV
# Please modify the line if OpenCV in your borad is installed at a different location
LDFLAGS   =   -L /opt/opencv/lib/ -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_imgcodecs
# LDFLAGS   = $(shell pkg-config --libs opencv)
# linking libraries of DNNDK 
LDFLAGS   +=  -lhineon -ln2cube -lpthread -ldputils

CUR_DIR   =   $(shell pwd)
SRC       =   $(CUR_DIR)/src
BUILD     =   $(CUR_DIR)/build
MODEL	  =   $(CUR_DIR)/model
VPATH     =   $(SRC)
SUDO      =   sudo
DST_DIR   =   $(CUR_DIR)/TGIIF/libraries/
#DST_DIR   =   /home/xilinx/jupyter_notebooks/dac_2018/TGIIF/libraries/

CFLAGS    :=  -O3 -mcpu=cortex-a9 -mfloat-abi=hard -mfpu=neon -Wall -Wpointer-arith -std=c++11 -ffast-math -I /opt/opencv/include/ -fPIC -shared -rdynamic
 
all: $(BUILD) $(PROJECT)
 
$(PROJECT): $(OBJ)
	$(CXX) $(CFLAGS) $(addprefix $(BUILD)/, $^) -o $@ $(LDFLAGS)
 
%.o : %.cc
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@

%.o : %.cpp
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@

%.elf : 
	cp $(MODEL)/$@ $(BUILD)/$@ 

clean:
	$(RM) -r $(BUILD)
	$(RM) $(PROJECT)

$(BUILD) : 
	-mkdir -p $@

copy:
	$(SUDO) cp $(CUR_DIR)/$(PROJECT) $(DST_DIR)
