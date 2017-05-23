CXX=/usr/bin/g++

INCLUDE_DIRS=$(OPENCV_DIR)/include
LIBRARY_DIRS=$(OPENCV_DIR)/lib

LIBRARIES=-lopencv_highgui -lopencv_imgcodecs -lopencv_calib3d -lopencv_core -lopencv_imgproc -lopencv_ml -lopencv_features2d -lopencv_flann -fopenmp

INCLUDE_FLAGS=$(foreach dir, $(INCLUDE_DIRS), -I$(dir))
LIBRARY_FLAGS=$(foreach dir, $(LIBRARY_DIRS), -L$(dir)) $(LIBRARIES)

CXX_FLAGS=-D_LINUX_MAC -D_OPENCV -O3 $(INCLUDE_FLAGS) 

OPT_FLOW_LIB_SRCS=GaussianPyramid.cpp OpticalFlow.cpp CVOpticalFlow.cpp Stochastic.cpp 



all: opt_flow.out

opt_flow_lib: $(OPT_FLOW_LIB_SRCS)
	 $(CXX) $(CXX_FLAGS) $(OPT_FLOW_LIB_SRCS) -c
	 ar rvs CVOpticalFlow.a CVOpticalFlow.o GaussianPyramid.o Stochastic.o OpticalFlow.o

opt_flow.out: xuanluo.cpp opt_flow_lib  
	$(CXX) $(CXX_FLAGS) $(LIBRARY_FLAGS) $< CVOpticalFlow.o CVOpticalFlow.a -o $@ 

aek_opt_flow_backup.out: aek.cpp opt_flow_lib
	$(CXX) $(CXX_FLAGS) $(LIBRARY_FLAGS) $< CVOpticalFlow.o CVOpticalFlow.a -o $@ 

clean:
	rm *.out
