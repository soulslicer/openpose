#ifndef OPENPOSE_TRACKER_HPP
#define OPENPOSE_TRACKER_HPP

#include <iostream>

#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// OpenPose dependencies
#include <caffe/caffe.hpp>
#include <openpose/gpu/cuda.hpp>
#include <dirent.h>

template<typename Dtype>
void matToCaffe(Dtype* caffeImg, const cv::Mat& imgAug){
    const int imageAugmentedArea = imgAug.rows * imgAug.cols;
    auto* uCharPtrCvMat = (unsigned char*)(imgAug.data);
    for (auto y = 0; y < imgAug.rows; y++)
    {
        const auto yOffset = y*imgAug.cols;
        for (auto x = 0; x < imgAug.cols; x++)
        {
            const auto xyOffset = yOffset + x;
            // const cv::Vec3b& bgr = imageAugmented.at<cv::Vec3b>(y, x);
            auto* bgr = &uCharPtrCvMat[3*xyOffset];
            caffeImg[xyOffset] = (bgr[0] - 128) / 256.0;
            caffeImg[xyOffset + imageAugmentedArea] = (bgr[1] - 128) / 256.0;
            caffeImg[xyOffset + 2*imageAugmentedArea] = (bgr[2] - 128) / 256.0;
        }
    }
}

bool nonzeropair(std::vector<float> p1,std::vector<float> p2, cv::Mat& img, std::vector<int>& color){
    if(p1[2] >= 0.05 && p2[2] >= 0.05){
        cv::line(img, cv::Point(p1[0], p1[1]), cv::Point(p2[0], p2[1]), cv::Scalar(color[0], color[1], color[2]), 2);
    }
}

void draw_lines_coco_21(cv::Mat& img, op::Array<float> mPoses, std::vector<int> mid, std::map<std::string, int>& MAPPING, std::vector<std::vector<int>>& colors){
    for(int i=0; i<mPoses.getSize()[0]; i++){
        auto nose_point = {mPoses.at({i,MAPPING["NOSE"],0}), mPoses.at({i,MAPPING["NOSE"],1}), mPoses.at({i,MAPPING["NOSE"],2})};
        auto neck_point = {mPoses.at({i,MAPPING["NECK"],0}), mPoses.at({i,MAPPING["NECK"],1}), mPoses.at({i,MAPPING["NECK"],2})};
        auto rshoulder_point = {mPoses.at({i,MAPPING["RSHOULDER"],0}), mPoses.at({i,MAPPING["RSHOULDER"],1}), mPoses.at({i,MAPPING["RSHOULDER"],2})};
        auto relbow_point = {mPoses.at({i,MAPPING["RELBOW"],0}), mPoses.at({i,MAPPING["RELBOW"],1}), mPoses.at({i,MAPPING["RELBOW"],2})};
        auto rwrist_point = {mPoses.at({i,MAPPING["RWRIST"],0}), mPoses.at({i,MAPPING["RWRIST"],1}), mPoses.at({i,MAPPING["RWRIST"],2})};
        auto lshoulder_point = {mPoses.at({i,MAPPING["LSHOULDER"],0}), mPoses.at({i,MAPPING["LSHOULDER"],1}), mPoses.at({i,MAPPING["LSHOULDER"],2})};
        auto lwrist_point = {mPoses.at({i,MAPPING["LWRIST"],0}), mPoses.at({i,MAPPING["LWRIST"],1}), mPoses.at({i,MAPPING["LWRIST"],2})};
        auto rhip_point = {mPoses.at({i,MAPPING["RHIP"],0}), mPoses.at({i,MAPPING["RHIP"],1}), mPoses.at({i,MAPPING["RHIP"],2})};
        auto rknee_point = {mPoses.at({i,MAPPING["RKNEE"],0}), mPoses.at({i,MAPPING["RKNEE"],1}), mPoses.at({i,MAPPING["RKNEE"],2})};
        auto rankle_point = {mPoses.at({i,MAPPING["RANKLE"],0}), mPoses.at({i,MAPPING["RANKLE"],1}), mPoses.at({i,MAPPING["RANKLE"],2})};
        auto lhip_point = {mPoses.at({i,MAPPING["LHIP"],0}), mPoses.at({i,MAPPING["LHIP"],1}), mPoses.at({i,MAPPING["LHIP"],2})};
        auto lknee_point = {mPoses.at({i,MAPPING["LKNEE"],0}), mPoses.at({i,MAPPING["LKNEE"],1}), mPoses.at({i,MAPPING["LKNEE"],2})};
        auto lankle_point = {mPoses.at({i,MAPPING["LANKLE"],0}), mPoses.at({i,MAPPING["LANKLE"],1}), mPoses.at({i,MAPPING["LANKLE"],2})};
        auto reye_point = {mPoses.at({i,MAPPING["REYE"],0}), mPoses.at({i,MAPPING["REYE"],1}), mPoses.at({i,MAPPING["REYE"],2})};
        auto leye_point = {mPoses.at({i,MAPPING["LEYE"],0}), mPoses.at({i,MAPPING["LEYE"],1}), mPoses.at({i,MAPPING["LEYE"],2})};
        auto rear_point = {mPoses.at({i,MAPPING["REAR"],0}), mPoses.at({i,MAPPING["REAR"],1}), mPoses.at({i,MAPPING["REAR"],2})};
        auto lear_point = {mPoses.at({i,MAPPING["LEAR"],0}), mPoses.at({i,MAPPING["LEAR"],1}), mPoses.at({i,MAPPING["LEAR"],2})};
        auto top_point = {mPoses.at({i,MAPPING["TOP"],0}), mPoses.at({i,MAPPING["TOP"],1}), mPoses.at({i,MAPPING["TOP"],2})};
        auto midhip_point = {mPoses.at({i,MAPPING["LOWERABS"],0}), mPoses.at({i,MAPPING["LOWERABS"],1}), mPoses.at({i,MAPPING["LOWERABS"],2})};
        auto realneck_point = {mPoses.at({i,MAPPING["REALNECK"],0}), mPoses.at({i,MAPPING["REALNECK"],1}), mPoses.at({i,MAPPING["REALNECK"],2})};
        auto lelbow_point = {mPoses.at({i,MAPPING["LELBOW"],0}), mPoses.at({i,MAPPING["LELBOW"],1}), mPoses.at({i,MAPPING["LELBOW"],2})};

        auto color = colors[mid[i] % colors.size()];
        nonzeropair(nose_point, neck_point, img, color);
        nonzeropair(leye_point, nose_point, img, color);
        nonzeropair(reye_point, nose_point, img, color);
        nonzeropair(rear_point, reye_point, img, color);
        nonzeropair(lear_point, leye_point, img, color);
        nonzeropair(neck_point, lshoulder_point, img, color);
        nonzeropair(neck_point, rshoulder_point, img, color);
        nonzeropair(relbow_point, rshoulder_point, img, color);
        nonzeropair(relbow_point, rwrist_point, img, color);
        nonzeropair(lelbow_point, lshoulder_point, img, color);
        nonzeropair(lelbow_point, lwrist_point, img, color);
        nonzeropair(midhip_point, lhip_point, img, color);
        nonzeropair(midhip_point, rhip_point, img, color);
        nonzeropair(midhip_point, neck_point, img, color);
        nonzeropair(rknee_point, rhip_point, img, color);
        nonzeropair(lknee_point, lhip_point, img, color);
        nonzeropair(rknee_point, rankle_point, img, color);
        nonzeropair(lknee_point, lankle_point, img, color);
        nonzeropair(top_point, realneck_point, img, color);
    }
}

#endif

