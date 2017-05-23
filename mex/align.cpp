// Rule: g++ -D_LINUX_MAC -D_OPENCV -I/usr/include/opencv -I/usr/local/include/opencv -L/usr/local/lib align.cpp CVOpticalFlow.a -fopenmp -lopencv_highgui -lopencv_calib3d -lopencv_core -lopencv_imgproc -lopencv_ml -lopencv_features2d -lopencv_flann -o align
#include "CVOpticalFlow.h"
#include <iostream>

using namespace std;
using namespace cv;
#define e3 at<Vec3b>

string base = "/usr/local/google/home/supasorn/focalstackdata/";
int n = 40;
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

vector<Mat> imgs;
struct flow {
  Mat x, y;
};
vector<flow> v;
void loadFiles() {
  char tmp[256];
  Mat im;
  FILE *fi;
  imgs.resize(n);
  v.resize(n);
  //string set = "_reg01";
  string set = "2";
  for (int i = 0; i < n; i++) {
    sprintf(tmp, (base + "vis" + set + "/im%02d.png").c_str(), i);
    imgs[i] = imread(tmp);
    imgs[i].convertTo(imgs[i], CV_64FC3, 1.0/255, 0);

    if (i > 0) {
      sprintf(tmp, (base + "vis" + set + "/vx%02d.bin").c_str(), i);
      fi = fopen(tmp, "rb");
      v[i].x = Mat(imgs[i].size(), CV_64F);
      fread(&v[i].x.at<double>(0, 0), sizeof(double), v[i].x.rows * v[i].x.cols, fi);
      fclose(fi);

      sprintf(tmp, (base + "vis" + set + "/vy%02d.bin").c_str(), i);
      fi = fopen(tmp, "rb");
      v[i].y = Mat(imgs[i].size(), CV_64F);
      fread(&v[i].y.at<double>(0, 0), sizeof(double), v[i].y.rows * v[i].y.cols, fi);
      fclose(fi);
    }
  }
}

void clip(int &a, int lo, int hi) {
  a = (a < lo) ? lo : (a>=hi ? hi-1: a);
}

flow composeFlow(flow &v0, flow &v1) {
  flow out;
  out.x = Mat(v0.x.size(), CV_64F);
  out.y = Mat(v0.y.size(), CV_64F);
  for (int r = 0; r < v0.x.rows; r++) {
    for (int c = 0; c < v0.x.cols; c++) {
      double dr = v0.y.at<double>(r, c) + r;
      double dc = v0.x.at<double>(r, c) + c;
      int r0 = dr, r1 = r0 + 1;
      int c0 = dc, c1 = c0 + 1;

      clip(r0, 0, v0.x.rows);
      clip(r1, 0, v0.x.rows);
      clip(c0, 0, v0.x.cols);
      clip(c1, 0, v0.x.cols);

      double tr = dr - r0;
      double tc = dc - c0;
      double ptr00, ptr01, ptr10, ptr11;
      ptr00 = v1.x.at<double>(r0, c0);
      ptr01 = v1.x.at<double>(r0, c1);
      ptr10 = v1.x.at<double>(r1, c0);
      ptr11 = v1.x.at<double>(r1, c1);
      double nx = ((1-tr) * (tc * ptr01 + (1-tc) * ptr00) + tr * (tc * ptr11 + (1-tc) * ptr10));
      ptr00 = v1.y.at<double>(r0, c0);
      ptr01 = v1.y.at<double>(r0, c1);
      ptr10 = v1.y.at<double>(r1, c0);
      ptr11 = v1.y.at<double>(r1, c1);
      double ny = ((1-tr) * (tc * ptr01 + (1-tc) * ptr00) + tr * (tc * ptr11 + (1-tc) * ptr10));

      out.y.at<double>(r, c) = ny + v0.y.at<double>(r, c);
      out.x.at<double>(r, c) = nx + v0.x.at<double>(r, c);
      //out.y.at<double>(r, c) = ny + dr;
      //out.x.at<double>(r, c) = nx + dc;
    }
  }
  return out;
}

vector<flow> chains;
vector<Mat> aligned;

//vector<flow> chains2;
//vector<Mat> 
void computeFlowChain() {
  chains.resize(n);
  chains[n-1] = v[n-1];
  for (int i = n - 2; i >= 1; i--) {
    chains[i] = composeFlow(chains[i+1], v[i]);
    //chains[i] = v[i];
  }
}

void mouse(int event, int x, int y, int flags, void *param) {
  if (event == CV_EVENT_MOUSEMOVE) {

    int id = y * n / aligned[0].rows;
    printf("%d\n", id);
    imshow("a", imgs[id]);
    moveWindow("a", 100, 100);
    imshow("b", aligned[id]);
    moveWindow("b", 100, 500);

  }
}

vector<Mat> contrast;
void allFocus() {
  Mat depth(aligned[0].size(), CV_64F);
  contrast.resize(n);
  int size = 3;
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    contrast[i] = Mat(aligned[i].size(), CV_64F);
    printf("Computing contrast %d\n", i);
    for (int j = 0; j < contrast[i].rows; j++) {
      for (int k = 0; k < contrast[i].cols; k++) {
        int count = 0;
        double diff = 0;
        for (int l = -size; l <= size; l++) {
          for (int m = -size; m <= size; m++) {
            int nr = j + l, nc = k + m;
            if (nr >=0 && nr < contrast[i].rows && nc >= 0 && nc < contrast[i].cols) {
              count ++;
              double d0 = aligned[i].at<Vec3d>(nr, nc)[0] - aligned[i].at<Vec3d>(j, k)[0];
              double d1 = aligned[i].at<Vec3d>(nr, nc)[1] - aligned[i].at<Vec3d>(j, k)[1];
              double d2 = aligned[i].at<Vec3d>(nr, nc)[2] - aligned[i].at<Vec3d>(j, k)[2];
              diff += d0 * d0 + d1 * d1 + d2 * d2;
            }
          }
        }
        diff /= count;
        contrast[i].at<double>(j, k) = diff;
      }
    }
  }

  Mat allFoc(contrast[0].size(), CV_64FC3);
  
  for (int j = 0; j < contrast[0].rows; j++) {
    for (int k = 0; k < contrast[0].cols; k++) {
      double mx = 0;
      int id = 0; 
      for (int i = 0; i < n; i++) {
        if (contrast[i].at<double>(j, k) > mx) {
          mx = contrast[i].at<double>(j, k);
          id = i;
        }
      }
      allFoc.at<Vec3d>(j, k) = aligned[id].at<Vec3d>(j, k);
      depth.at<double>(j, k) = 1.0 * id / (n-1);
    }
  }
  imshow("All focus", allFoc);
  imshow("depth", depth);
  waitKey(0);
}
int main() {

  double alpha = 0.03;
  double ratio = 0.85;
  int minWidth = 20;
  int nOuterFPIterations = 4;
  int nInnerFPIterations = 1;
  int nSORIterations = 40;

  char tmp[256];
  double scale = 0.5;

  loadFiles();
  printf("Computing chain\n");
  computeFlowChain();
  printf("Done\n");
  aligned.resize(n);
  aligned[n-1] = imgs[n-1].clone();
  for (int i = n-2; i >= 0; i--) {
    Mat out;
    CVOpticalFlow::warp(aligned[i], imgs[i], chains[i+1].x, chains[i+1].y);
  }
  allFocus();

  imshow("a", imgs[n-1]);
  moveWindow("a", 100, 100);
  imshow("b", aligned[n-1]);
  moveWindow("b", 100, 500);
  setMouseCallback("a", mouse);
  setMouseCallback("b", mouse);
  waitKey(0);



}

