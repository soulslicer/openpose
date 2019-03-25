#include <limits> // std::numeric_limits
#include <openpose/gpu/cuda.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/pose/poseExtractorCaffeStaf.hpp>
#include <iostream>

namespace op
{
    PoseExtractorCaffeStaf::PoseExtractorCaffeStaf(
        const PoseModel poseModel, const std::string& modelFolder, const int gpuId,
        const std::vector<HeatMapType>& heatMapTypes, const ScaleMode heatMapScaleMode, const bool addPartCandidates,
        const bool maximizePositives, const std::string& protoTxtPath, const std::string& caffeModelPath,
        const float upsamplingRatio, const bool enableNet, const bool enableGoogleLogging) :
        PoseExtractorCaffe{poseModel, modelFolder, gpuId, heatMapTypes, heatMapScaleMode, addPartCandidates,
        maximizePositives, protoTxtPath, caffeModelPath, upsamplingRatio, enableNet, enableGoogleLogging}
    {
log("RUNNING PoseExtractorCaffeStaf::PoseExtractorCaffeStaf");
    }

    PoseExtractorCaffeStaf::~PoseExtractorCaffeStaf()
    {
    }

    void PoseExtractorCaffeStaf::addCaffeNetOnThread()
    {
        this->spNets.emplace_back(
            std::make_shared<NetCaffe>(
                this->mModelFolder + "pose/body_25b_video/pose_deploy.prototxt",
                this->mModelFolder + "pose/body_25b_video/pose_iter_XXXXXX.caffemodel",
                this->mGpuId, this->mEnableGoogleLogging));

        // Initialize
        this->spNets.back()->initializationOnThread();


        this->spCaffeNetOutputBlobs.emplace_back((this->spNets.back().get())->getOutputBlobArray());

        // Add other outputs as reference
        this->mLastPafBlobs.emplace_back((this->spNets.back().get())->getBlobArray("Mconv7_stage1_L2"));
        this->mLastHmBlobs.emplace_back((this->spNets.back().get())->getBlobArray("Mconv7_stage2_L1"));
        this->mLastTafBlobs.emplace_back((this->spNets.back().get())->getBlobArray("Mconv7_stage3_L4"));
        this->mLastFmBlobs.emplace_back((this->spNets.back().get())->getBlobArray("conv4_4_CPM"));
    }

    void PoseExtractorCaffeStaf::netInitializationOnThread()
    {
        try
        {
            // Add Caffe Net
            addCaffeNetOnThread();

            // Initialize blobs
            spHeatMapsBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};
            spPeaksBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<int> reduceSize(const std::vector<int>& outputSize, const std::vector<int>& inputSize, int stride=8)
    {
        std::vector<int> finalSize(outputSize);
        finalSize[2] = (inputSize[2]/stride);
        finalSize[3] = (inputSize[3]/stride);
        return finalSize;
    }

    std::vector<ArrayCpuGpu<float>*> arraySharedToPtr2(
        const std::vector<std::shared_ptr<ArrayCpuGpu<float>>>& caffeNetOutputBlob)
    {
        try
        {
            // Prepare spCaffeNetOutputBlobss
            std::vector<ArrayCpuGpu<float>*> caffeNetOutputBlobs(caffeNetOutputBlob.size());
            for (auto i = 0u ; i < caffeNetOutputBlobs.size() ; i++)
                caffeNetOutputBlobs[i] = caffeNetOutputBlob[i].get();
            return caffeNetOutputBlobs;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    inline void reshapePoseExtractorCaffe(
        std::shared_ptr<ResizeAndMergeCaffe<float>>& resizeAndMergeCaffe,
        std::shared_ptr<NmsCaffe<float>>& nmsCaffe,
        std::shared_ptr<BodyPartConnectorCaffe<float>>& bodyPartConnectorCaffe,
        std::shared_ptr<MaximumCaffe<float>>& maximumCaffe,
        std::vector<std::shared_ptr<ArrayCpuGpu<float>>>& caffeNetOutputBlobsShared,
        std::shared_ptr<ArrayCpuGpu<float>>& heatMapsBlob, std::shared_ptr<ArrayCpuGpu<float>>& peaksBlob,
        std::shared_ptr<ArrayCpuGpu<float>>& maximumPeaksBlob, const float scaleInputToNetInput,
        const PoseModel poseModel, const int gpuId, const float upsamplingRatio)
    {
        try
        {
            const auto netDescreaseFactor = (
                upsamplingRatio <= 0.f ? getPoseNetDecreaseFactor(poseModel) : upsamplingRatio);
            // HeatMaps extractor blob and layer
            // Caffe modifies bottom - Heatmap gets resized
            const auto caffeNetOutputBlobs = arraySharedToPtr2(caffeNetOutputBlobsShared);
            resizeAndMergeCaffe->Reshape(
                caffeNetOutputBlobs, {heatMapsBlob.get()},
                netDescreaseFactor, 1.f/scaleInputToNetInput, true, gpuId);
            // Pose extractor blob and layer
            nmsCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()}, getPoseMaxPeaks(),
                              getPoseNumberBodyParts(poseModel), gpuId);
            // Pose extractor blob and layer
            bodyPartConnectorCaffe->Reshape({heatMapsBlob.get(), peaksBlob.get()}, gpuId);
            // Cuda check
            #ifdef USE_CUDA
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractorCaffeStaf::forwardPass(
        const std::vector<Array<float>>& inputNetData, const Point<int>& inputDataSize,
        const std::vector<double>& scaleInputToNetInputs, const Array<float>& poseNetOutput)
    {
        try
        {
            // Sanity checks
            if (inputNetData.empty())
                error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
            for (const auto& inputNetDataI : inputNetData)
                if (inputNetDataI.empty())
                    error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
            if (inputNetData.size() != scaleInputToNetInputs.size())
                error("Size(inputNetData) must be same than size(scaleInputToNetInputs).",
                      __LINE__, __FUNCTION__, __FILE__);
            if (poseNetOutput.empty() != mEnableNet)
            {
                const std::string errorMsg = ". Either use OpenPose default network (`--body 1`) or fill the"
                    " `poseNetOutput` argument (only 1 of those 2, not both).";
                if (poseNetOutput.empty())
                    error("The argument poseNetOutput cannot be empty if mEnableNet is true" + errorMsg,
                          __LINE__, __FUNCTION__, __FILE__);
                else
                    error("The argument poseNetOutput is not empty and you have also explicitly chosen to run"
                          " the OpenPose network" + errorMsg, __LINE__, __FUNCTION__, __FILE__);
            }

            // Resize std::vectors if required
            const auto numberScales = inputNetData.size();
            mNetInput4DSizes.resize(numberScales);

            // Add to Net
            while (spNets.size() < numberScales){
                addCaffeNetOnThread();
            }

            // Reshape
            for (auto i = 0u ; i < inputNetData.size(); i++)
            {
                const auto changedVectors = !vectorsAreEqual(
                    mNetInput4DSizes.at(i), inputNetData[i].getSize());
                if (changedVectors)
                {
                    mNetInput4DSizes.at(i) = inputNetData[i].getSize();

                    // First reshape net
                    auto inputSize = inputNetData[i].getSize();
                    spNets.at(i)->reshape(inputNetData[i].getSize(), "image", 0);
                    spNets.at(i)->reshape(reduceSize(spNets.at(i)->shape("last_paf"), inputSize), "last_paf", 0);
                    spNets.at(i)->reshape(reduceSize(spNets.at(i)->shape("last_hm"), inputSize), "last_hm", 0);
                    spNets.at(i)->reshape(reduceSize(spNets.at(i)->shape("last_taf"), inputSize), "last_taf", 0);
                    spNets.at(i)->reshape(reduceSize(spNets.at(i)->shape("last_fm"), inputSize), "last_fm", 1);

                    // Reshape Other
                    reshapePoseExtractorCaffe(
                        spResizeAndMergeCaffe, spNmsCaffe, spBodyPartConnectorCaffe,
                        spMaximumCaffe, spCaffeNetOutputBlobs, spHeatMapsBlob,
                        spPeaksBlob, spMaximumPeaksBlob, 1.f, mPoseModel,
                        mGpuId, mUpsamplingRatio);
                }
            }

            // CUDA Copy Stuff
            for (auto i = 0u ; i < inputNetData.size(); i++)
            {

                // Need a way to copy netCopy(

//                auto* gpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_gpu_data();
//                cudaMemcpy(gpuImagePtr, inputNetData[i].getConstPtr(), inputNetData[i].getVolume() * sizeof(float),
//                           cudaMemcpyHostToDevice);
            }

        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
