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

Mat dx(Mat in) {
  double f[5] = {1, -8, 0, 8, -1};
  for (int i = 0; i < 5; i++)
    f[i] /= 12;

  Mat out = Mat::zeros(in.size(), CV_64F);
  for (int i = 0; i < in.rows; i++) {
    for (int j = 2; j < in.cols - 2; j++) {
      for (int k = 0; k < 5; k++) {
        out.at<double>(i, j) += in.at<double>(i, j + k - 2) * f[k];
      }
    }
  }
  return out;
}

Mat dy(Mat in) {
  double f[5] = {1, -8, 0, 8, -1};
  for (int i = 0; i < 5; i++)
    f[i] /= 12;

  Mat out = Mat::zeros(in.size(), CV_64F);
  for (int i = 2; i < in.rows - 2; i++) {
    for (int j = 0; j < in.cols; j++) {
      for (int k = 0; k < 5; k++) {
        out.at<double>(i, j) += in.at<double>(i + k - 2, j) * f[k];
      }
    }
  }
  return out;
}


void testND() {
  double alpha = 0.03;
  double ratio = 0.85;
  int minWidth = 20;
  int nOuterFPIterations = 4;
  int nInnerFPIterations = 1;
  int nSORIterations = 40;

  Mat im1 = imread("../bush2.png");
  Mat im2 = imread("../bush1.png");
  Mat im1g, im2g;
  cvtColor(im1, im1g, CV_BGR2GRAY);
  cvtColor(im2, im2g, CV_BGR2GRAY);
  im1.convertTo(im1, CV_64FC3, 1.0/255, 0);
  im2.convertTo(im2, CV_64FC3, 1.0/255, 0);
  im1g.convertTo(im1g, CV_64FC3, 1.0/255, 0);
  im2g.convertTo(im2g, CV_64FC3, 1.0/255, 0);

  Mat vx, vy, warp;

  imshow("im1", im1);
  moveWindow("im1", 300, 100);

  imshow("im2", im2);
  moveWindow("im2", 100, 100);

  vector<Mat> im1s;
  vector<Mat> im2s;
  split(im1, im1s);
  split(im2, im2s);


  vector<Mat> r1;
  vector<Mat> r2;

  r1.push_back(im1s[1] - im1s[0]);
  r1.push_back(im1s[1] - im1s[2]);
  r1.push_back(dx(im1g));
  r1.push_back(dy(im1g));
  r1.push_back(im1g);

  r2.push_back(im2s[1] - im2s[0]);
  r2.push_back(im2s[1] - im2s[2]);
  r2.push_back(dx(im2g));
  r2.push_back(dy(im2g));
  r2.push_back(im2g);


  CVOpticalFlow::findFlowND(vx, vy, warp, r1, r2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);

  Mat mywarp;
  CVOpticalFlow::warp(mywarp, im2, vx, vy);
  imshow("a", mywarp);
  imshow("flow", CVOpticalFlow::showFlow(vx, vy));
  moveWindow("flow", 700, 100);
  waitKey(0);
}


int main() {

  //testND();
  double alpha = 0.03;
  double ratio = 0.85;
  int minWidth = 20;
  int nOuterFPIterations = 4;
  int nInnerFPIterations = 1;
  int nSORIterations = 40;

  printf("standard\n");
  Mat im1 = imread("../bush2.png");
  Mat im2 = imread("../bush1.png");
  //Mat im1 = imread("../car1.jpg");
  //Mat im2 = imread("../car2.jpg");
  im1.convertTo(im1, CV_64FC3, 1.0/255, 0);
  im2.convertTo(im2, CV_64FC3, 1.0/255, 0);
  Mat vx, vy, warp;

  //Mat im1gray;
  //toGray(im1gray, im1);
  //Mat im2gray;
  //toGray(im2gray, im2);
  imshow("im1", im1);
  moveWindow("im1", 300, 100);

  imshow("im2", im2);
  moveWindow("im2", 100, 100);
  for (double alp = 0.005; 0 && alp <= 0.1; alp += 0.005) {
    printf("alpha = %f\n", alp);
    CVOpticalFlow::findFlow(vx, vy, warp, im1, im2, alp, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
    char tmp[256];
    sprintf(tmp, "bushalpha_a_%04d.png", int(alp * 1000));
    imwrite(tmp, warp);
    imshow("warp", warp);
    moveWindow("warp", 500, 100);
  }
  CVOpticalFlow::findFlow(vx, vy, warp, im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);

  imshow("warp", warp);
  moveWindow("warp", 500, 100);
  //waitKey(0);
  Mat mywarp;
  //Mat tm;
  //to3(tm, im2gray);
  CVOpticalFlow::warp(mywarp, im2, vx, vy);
  imshow("a", mywarp);
  //waitKey(0);
  imshow("flow", CVOpticalFlow::showFlow(vx, vy));
  moveWindow("flow", 700, 100);
  waitKey(0);

}

