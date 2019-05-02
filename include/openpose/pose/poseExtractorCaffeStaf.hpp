#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_STAF_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_STAF_HPP

#include <openpose/core/common.hpp>
// #include <openpose/net/bodyPartConnectorCaffe.hpp>
// #include <openpose/net/maximumCaffe.hpp>
// #include <openpose/net/netCaffe.hpp>
// #include <openpose/net/netOpenCv.hpp>
// #include <openpose/net/nmsCaffe.hpp>
// #include <openpose/net/resizeAndMergeCaffe.hpp>
// #include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseExtractorCaffe.hpp>
#include <openpose/pose/poseTracker.hpp>

namespace op
{
    class OP_API PoseExtractorCaffeStaf : public PoseExtractorCaffe
    {
    public:
        PoseExtractorCaffeStaf(
            const PoseModel poseModel, const std::string& modelFolder, const int gpuId,
            const std::vector<HeatMapType>& heatMapTypes = {},
            const ScaleMode heatMapScaleMode = ScaleMode::ZeroToOne,
            const bool addPartCandidates = false, const bool maximizePositives = false,
            const std::string& protoTxtPath = "", const std::string& caffeModelPath = "",
            const float upsamplingRatio = 0.f, const bool enableNet = true,
            const bool enableGoogleLogging = true);

        virtual ~PoseExtractorCaffeStaf();

        virtual void netInitializationOnThread();

        void addCaffeNetOnThread();

        /**
         * @param poseNetOutput If it is not empty, OpenPose will not run its internal body pose estimation network
         * and will instead use this data as the substitute of its network. The size of this element must match the
         * size of the output of its internal network, or it will lead to core dumped (segmentation) errors. You can
         * modify the pose estimation flags to match the dimension of both elements (e.g., `--net_resolution`,
         * `--scale_number`, etc.).
         */
        virtual void forwardPass(
            const std::vector<Array<float>>& inputNetData, const Point<int>& inputDataSize,
            const std::vector<double>& scaleInputToNetInputs = {1.f},
            const Array<float>& poseNetOutput = Array<float>{});

        std::vector<std::shared_ptr<ArrayCpuGpu<float>>> mCurrPafBlobs;
        std::vector<std::shared_ptr<ArrayCpuGpu<float>>> mCurrHmBlobs;
        std::vector<std::shared_ptr<ArrayCpuGpu<float>>> mCurrTafBlobs;
        std::vector<std::shared_ptr<ArrayCpuGpu<float>>> mCurrFmBlobs;

        std::vector<std::shared_ptr<ArrayCpuGpu<float>>> mLastPafBlobs;
        std::vector<std::shared_ptr<ArrayCpuGpu<float>>> mLastHmBlobs;
        std::vector<std::shared_ptr<ArrayCpuGpu<float>>> mLastTafBlobs;
        std::vector<std::shared_ptr<ArrayCpuGpu<float>>> mLastFmBlobs;

        std::shared_ptr<ArrayCpuGpu<float>> spTafsBlob;

        std::unique_ptr<PoseTracker> mPoseTracker;

        DELETE_COPY(PoseExtractorCaffeStaf);
    };
}

#endif // OPENPOSE_POSE_POSE_EXTRACTOR_CAFFE_STAF_HPP
