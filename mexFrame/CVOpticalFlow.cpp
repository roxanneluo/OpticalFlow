#include "CVOpticalFlow.h"

Mat CVOpticalFlow::showFlow(Mat &vxi, Mat &vyi) {
  Mat vx = vxi.clone();
  Mat vy = vyi.clone();
  int thresh = 1e9;
  Mat out = Mat(vx.size(), CV_64FC3);
  double maxrad = -1;
  for (int i=0; i<out.rows; i++) {
    for (int j=0; j<out.cols; j++) {
      if (fabs(vx.at<double>(i, j)) > thresh) 
        vx.at<double>(i, j) = 0;
      if (fabs(vy.at<double>(i, j)) > thresh) 
        vy.at<double>(i, j) = 0;
      double rad = vx.at<double>(i, j) * vx.at<double>(i, j) + vy.at<double>(i, j) * vy.at<double>(i, j);
      maxrad = max(maxrad, rad);
    }
  }
  maxrad = sqrt(maxrad);
  for (int i=0; i<out.rows; i++) {
    for (int j=0; j<out.cols; j++) {
      vx.at<double>(i, j) /= maxrad;
      vy.at<double>(i, j) /= maxrad;
      out.at<Vec3d>(i, j)[0] = vx.at<double>(i, j) * 0.5 + 0.5;
      out.at<Vec3d>(i, j)[1] = vy.at<double>(i, j) * 0.5 + 0.5;
      out.at<Vec3d>(i, j)[2] = 0;
    }
  }
  return out;
}
inline void CVOpticalFlow::clip(int &a, int lo, int hi) {
  a = (a < lo) ? lo : (a>=hi ? hi-1: a);
}

void CVOpticalFlow::bilinear(double *out, Mat &im, double r, double c, int channels) {
  int r0 = r, r1 = r+1;
  int c0 = c, c1 = c+1;
  clip(r0, 0, im.rows);
  clip(r1, 0, im.rows);
  clip(c0, 0, im.cols);
  clip(c1, 0, im.cols);

  double tr = r - r0;
  double tc = c - c0;
  for (int i=0; i<channels; i++) {
    double ptr00 = im.at<Vec3d>(r0, c0)[i];
    double ptr01 = im.at<Vec3d>(r0, c1)[i];
    double ptr10 = im.at<Vec3d>(r1, c0)[i];
    double ptr11 = im.at<Vec3d>(r1, c1)[i];
    out[i] = ((1-tr) * (tc * ptr01 + (1-tc) * ptr00) + tr * (tc * ptr11 + (1-tc) * ptr10));
  }
}

void CVOpticalFlow::warp(Mat &out, Mat &im, Mat &vx, Mat &vy) {
  if(im.type()!=CV_64FC3) {
    printf("Error: unsupported typed. Required CV_64FC3");
    return ;
  }
  out = Mat(im.size(), CV_64FC3);
  for (int i=0; i<out.rows; i++) {
    for (int j=0; j<out.cols; j++) {
      bilinear(&out.at<Vec3d>(i, j)[0], im, i+vy.at<double>(i, j), j+vx.at<double>(i, j), 3);
    }
  }

}

void CVOpticalFlow::findFlowND(Mat &vx, Mat &vy, Mat &warp, vector<Mat> &im1, vector<Mat> &im2, double alpha, double ratio, int minWidth, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations) {
  
  DImage iim1, iim2;
  DImage ivx, ivy, iwarp;
  iim1.fromMats(im1);
  iim2.fromMats(im2);
  OpticalFlow::Coarse2FineFlow(ivx, ivy, iwarp, iim1, iim2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
  vx = Mat(im1[0].size(), CV_64F);
  vy = Mat(im2[0].size(), CV_64F);
  memcpy(vx.data, ivx.pData, sizeof(double) * vx.rows * vx.cols);
  memcpy(vy.data, ivy.pData, sizeof(double) * vy.rows * vy.cols);
  //iwarp.toMat(warp);
}

void CVOpticalFlow::findFlow(Mat &vx, Mat &vy, Mat &warp, Mat &im1, Mat &im2, double alpha, double ratio, int minWidth, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations) {
  DImage iim1, iim2;
  DImage ivx, ivy, iwarp;
  iim1.fromMat(im1);
  iim2.fromMat(im2);
  OpticalFlow::Coarse2FineFlow(ivx, ivy, iwarp, iim1, iim2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
  vx = Mat(im1.size(), CV_64F);
  vy = Mat(im2.size(), CV_64F);
  memcpy(vx.data, ivx.pData, sizeof(double) * vx.rows * vx.cols);
  memcpy(vy.data, ivy.pData, sizeof(double) * vy.rows * vy.cols);
  iwarp.toMat(warp);
}
