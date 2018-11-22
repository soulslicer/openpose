#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/pose/renderPose.hpp>

namespace op
{
    bool nonzeropair(std::vector<float> p1,std::vector<float> p2, cv::Mat& img, std::vector<int>& color, float scale=1.0){
        if(p1[2] >= 0.05 && p2[2] >= 0.05){
            cv::line(img, cv::Point(p1[0]*scale, p1[1]*scale), cv::Point(p2[0]*scale, p2[1]*scale), cv::Scalar(color[0], color[1], color[2]), 3);
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

    std::map<std::string, int> COCO21_MAPPING = {
        {"NOSE", 0},
        {"NECK", 1},
        {"RSHOULDER", 2},
        {"RELBOW", 3},
        {"RWRIST", 4},
        {"LSHOULDER", 5},
        {"LELBOW", 6},
        {"LWRIST", 7},
        {"LOWERABS", 8},
        {"RHIP", 9},
        {"RKNEE", 10},
        {"RANKLE", 11},
        {"LHIP", 12},
        {"LKNEE", 13},
        {"LANKLE", 14},
        {"REYE", 15},
        {"LEYE", 16},
        {"REAR", 17},
        {"LEAR", 18},
        {"REALNECK", 19},
        {"TOP", 20},
    };
    std::vector<std::vector<int>> colors = {{255,0,0},{0,255,0},{0,0,255},{255,255,0},{0,255,255},{255,0,255},{128,255,0},{0,128,255},{128,0,255},{128,0,0},{0,128,0},{0,0,128},{50,50,0},{50,50,0},{0,50,128},{0,100,128},{100,30,128},{0,50,255},{50,255,128},{100,90,128},{0,90,255},{50,90,128},{255,0,0},{0,255,0},{0,0,255},{255,255,0},{0,255,255},{255,0,255},{128,255,0},{0,128,255},{128,0,255},{128,0,0},{0,128,0},{0,0,128},{50,50,0},{50,50,0},{0,50,128},{0,100,128},{100,30,128},{0,50,255},{50,255,128},{100,90,128},{0,90,255},{50,90,128}};


    void renderPoseKeypointsCpu(Array<float>& frameArray, const Array<float>& poseKeypoints, const PoseModel poseModel,
                                const float renderThreshold, const bool blendOriginalFrame, Array<long long> poseIds)
    {
        try
        {
            if (!frameArray.empty())
            {
                // Background
                if (!blendOriginalFrame)
                    frameArray.getCvMat().setTo(0.f); // [0-255]

                // Parameters
                const auto thicknessCircleRatio = 1.f/75.f;
                const auto thicknessLineRatioWRTCircle = 0.15f;
                const auto& pairs = getPoseBodyPartPairsRender(poseModel);
                const auto& poseScales = getPoseScales(poseModel);

                // My stuff
                if(poseModel == PoseModel::BODY_21A){
                    // Mapping

                    //std::cout << frame
                    //draw_lines_coco_21()
                    // Array<T> --> cv::Mat
                    auto frame = frameArray.getCvMat();

                    // Get frame channels
                    const auto width = frame.size[1];
                    const auto height = frame.size[0];
                    const auto area = width * height;
                    cv::Mat frameBGR(height, width, CV_32FC3, frame.data);

                    std::vector<int> mid(poseIds.getSize(0));
                    for(int i=0; i<mid.size(); i++) mid[i] = poseIds.at(i);

                    draw_lines_coco_21(frameBGR, poseKeypoints, mid, COCO21_MAPPING, colors);
                }

                // Render keypoints
                renderKeypointsCpu(frameArray, poseKeypoints, pairs, getPoseColors(poseModel), thicknessCircleRatio,
                                   thicknessLineRatioWRTCircle, poseScales, renderThreshold);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
