#ifndef OPENPOSE_TRACKING_MOSSE_HPP
#define OPENPOSE_TRACKING_MOSSE_HPP

#include <openpose/core/common.hpp>

namespace op
{

class Random{
public:
    std::random_device rd;
    std::mt19937 mt;
    std::uniform_real_distribution<double> dist;
    Random(double start=0., double end=1.){
        mt = std::mt19937(rd());
        dist = std::uniform_real_distribution<double>(start,end);
    }
    double getRand(){
        return dist(mt);
    }
};

class MOSSE{
public:
    // Params
    float rand_coef = 0.2;
    float lr = 0.125;
    float min_psr = 2.0;
    int loss_action = 0;

    // Internal Variables
    Random random;
    cv::Point2f pos;
    cv::Size size;
    cv::Mat win;
    cv::Mat G, H1, H2, H;
    cv::Mat last_img, last_resp;
    float psr;
    bool good;

    cv::Mat rnd_warp(const cv::Mat& a);
    cv::Mat preprocess(const cv::Mat& img);
    void negImag(cv::Mat& B);
    cv::Mat divSpec(const cv::Mat& A, const cv::Mat& B);
    void update_kernel();
    std::tuple<cv::Mat, cv::Point2f, float> correlate(const cv::Mat& img);
    void update(cv::Mat frame);
    void draw_state(cv::Mat& vis);
    MOSSE(cv::Mat frame, std::array<int, 4> rect);

    //void save(cv::Mat m, std::string path, int i){
    //    cv::Mat splitted[2] = {cv::Mat(m.size(),CV_32F),cv::Mat(m.size(),CV_32F)};
    //    cv::split(m, splitted);
    //    //cv::imwrite(path, splitted[i]*255);
    //}

};

}

#endif // OPENPOSE_TRACKING_LKPYRAMIDAL_HPP
