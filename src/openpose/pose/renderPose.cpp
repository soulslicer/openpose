#include <openpose/pose/renderPose.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>

namespace op
{
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
                const auto thicknessLineRatioWRTCircle = 0.75f;
                const auto& pairs = getPoseBodyPartPairsRender(poseModel);
                const auto& poseScales = getPoseScales(poseModel);

//                for(int i=0; i<poseIds.getSize(0); i++) op::log(std::to_string(poseIds.at(i)));
//                op::log("**");
//                op::log(std::to_string(poseIds.getSize(0)));

                // Render keypoints
                renderKeypointsCpu(frameArray, poseKeypoints, pairs, getPoseColors(poseModel), thicknessCircleRatio,
                                   thicknessLineRatioWRTCircle, poseScales, renderThreshold, poseIds);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
