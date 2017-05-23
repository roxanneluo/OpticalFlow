g++ -D_LINUX_MAC -D_OPENCV -I/homes/grail/supasorn/include -I/homes/grail/supasorn/include/opencv GaussianPyramid.cpp OpticalFlow.cpp CVOpticalFlow.cpp Stochastic.cpp -c 
ar rvs CVOpticalFlow.a CVOpticalFlow.o GaussianPyramid.o Stochastic.o OpticalFlow.o

