g++ -D_LINUX_MAC -D_OPENCV -I/projects/grail/supasorn2nb/local/include/ -I/projects/grail/supasorn2nb/local/include/opencv GaussianPyramid.cpp OpticalFlow.cpp CVOpticalFlow.cpp Stochastic.cpp -c 
ar rvs CVOpticalFlownomp.a CVOpticalFlow.o GaussianPyramid.o Stochastic.o OpticalFlow.o

