// Rule: g++ -D_LINUX_MAC -D_OPENCV -I/usr/include/opencv -I/usr/local/include/opencv -L/usr/local/lib aek.cpp CVOpticalFlow.o CVOpticalFlow.a -lopencv_highgui -lopencv_calib3d -lopencv_core -lopencv_imgproc -lopencv_ml -lopencv_features2d -lopencv_flann -o test
#include "CVOpticalFlow.h"
#include <iostream>

using namespace std;
using namespace cv;
#define e3 at<Vec3b>

void toGray(Mat &out, Mat &in) {
  out = Mat::zeros(in.size(), CV_64F);
  for (int i=0; i<in.rows; i++) {
    for (int j=0; j<in.cols; j++) {
      out.at<double>(i, j) = (in.at<Vec3d>(i, j)[0] * 0.114 + in.at<Vec3d>(i, j)[1] * 0.587 + in.at<Vec3d>(i, j)[2] * 0.299);
    }
  }
}

void to3(Mat &out, Mat &in) {
  out = Mat::zeros(in.size(), CV_64FC3);
  for (int i=0; i<in.rows; i++) {
    for (int j=0; j<in.cols; j++) {
      out.at<Vec3d>(i, j)[0] = in.at<double>(i, j);
      out.at<Vec3d>(i, j)[1] = in.at<double>(i, j);
      out.at<Vec3d>(i, j)[2] = in.at<double>(i, j);
    }
  }
}
int main() {

  double alpha = 0.03;
  double ratio = 0.85;
  int minWidth = 20;
  int nOuterFPIterations = 4;
  int nInnerFPIterations = 1;
  int nSORIterations = 40;
  string base = "/usr/local/google/home/supasorn/focalstackdata/";


  for (int i=39; i>=1; i--) {
    printf("%d\n", i);
    char tmp[256];
    sprintf(tmp, (base + "RGBZ_test%d.png").c_str(), i - 1);
    Mat im1 = imread(tmp);
    sprintf(tmp, (base + "RGBZ_test%d.png").c_str(), i);
    Mat im2 = imread(tmp);

    double scale = 0.5;
    resize(im1, im1, Size(), scale, scale);
    resize(im2, im2, Size(), scale, scale);

    if (i == 1) {
      sprintf(tmp, (base + "vis2/im%02d.png").c_str(), i - 1);
      imwrite(tmp, im1);
    }
    sprintf(tmp, (base + "vis2/im%02d.png").c_str(), i);
    imwrite(tmp, im2);

    im1.convertTo(im1, CV_64FC3, 1.0/255, 0);
    im2.convertTo(im2, CV_64FC3, 1.0/255, 0);

    Mat vx, vy, warp;
    CVOpticalFlow::findFlow(vx, vy, warp, im2, im1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);


    sprintf(tmp, (base + "vis2/im%02dw.png").c_str(), i);
    imwrite(tmp, warp);

    sprintf(tmp, (base + "vis2/vx%02d.bin").c_str(), i);
    FILE *fo = fopen(tmp, "wb");
    fwrite(&vx.at<double>(0, 0), sizeof(double), vx.rows * vx.cols, fo);
    fclose(fo);

    sprintf(tmp, (base + "vis2/vy%02d.bin").c_str(), i);
    fo = fopen(tmp, "wb");
    fwrite(&vy.at<double>(0, 0), sizeof(double), vy.rows * vy.cols, fo);
    fclose(fo);
    //imshow("warp", warp);
    //moveWindow("warp", 500, 100);
  }
  Mat im1 = imread(base + "RGBZ_test0.png");
  Mat im2 = imread(base + "RGBZ_test39.png");
  //Mat im1 = imread("../car1.jpg");
  //Mat im2 = imread("../car2.jpg");
  double scale = 0.5;
  resize(im1, im1, Size(), scale, scale);
  resize(im2, im2, Size(), scale, scale);

  imwrite((base + "vis/im1.png").c_str(), im1);
  imwrite((base + "vis/im2.png").c_str(), im2);

  im1.convertTo(im1, CV_64FC3, 1.0/255, 0);
  im2.convertTo(im2, CV_64FC3, 1.0/255, 0);
  printf("%d %d\n", im1.rows, im1.cols);
  Mat vx, vy, warp;

  //Mat im1gray;
  //toGray(im1gray, im1);
  //Mat im2gray;
  //toGray(im2gray, im2);
  printf("c\n");
  imshow("im1", im1);
  moveWindow("im1", 100, 100);

  printf("c\n");
  imshow("im2", im2);
  moveWindow("im2", 100, 100 + im1.rows);

  Mat bl1;
  GaussianBlur(im1, bl1, Size(13, 13), 0, 0);
  imshow("bl1", bl1);

  //CVOpticalFlow::findFlow(vx, vy, warp, im2, im1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
  CVOpticalFlow::findFlow(vx, vy, warp, im2, bl1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);

  imshow("warp", warp);
  moveWindow("warp", 500, 100);

  //char tmp[256];
  //sprintf(tmp, (base + "warp.png").c_str());
  //printf("saving to %s\n", tmp);
  imwrite((base + "vis/im3.png").c_str(), warp);
  imshow("warp", warp);
  moveWindow("warp", 500, 100);

  imshow("flow", CVOpticalFlow::showFlow(vx, vy));
  moveWindow("flow", 700, 100);

  CVOpticalFlow::warp(warp, im1, vx, vy);
  warp.convertTo(warp, CV_8UC3, 255, 0);
  imwrite((base + "vis/im4.png").c_str(), warp);

  //imshow("warp", warp);
  //moveWindow("warp", 500, 100);
  waitKey(0);

}

