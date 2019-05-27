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

#include <cuda.h>
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include <caffe/caffe.hpp>

//#include <openpose/flags.hpp>

namespace op
{
    void gpu_copy(std::shared_ptr<ArrayCpuGpu<float>> dst, std::shared_ptr<ArrayCpuGpu<float>> src)
    {
        size_t size = sizeof(float)*src->shape()[1]*src->shape()[2]*src->shape()[3];
        cudaMemcpy(dst->mutable_gpu_data(), src->gpu_data(), size, cudaMemcpyDeviceToDevice);
    }

    cv::Mat mat_from_blob(std::shared_ptr<ArrayCpuGpu<float>> src, int channel)
    {
        cv::Mat m(src->shape()[2], src->shape()[3], CV_32FC1, (float*)(src->cpu_data() + channel*src->shape()[2]*src->shape()[3]));
        return m;
    }

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
        if(mPoseModel == PoseModel::BODY_25B)
        {
            // Tracker
            this->mPoseTracker = std::unique_ptr<PoseTracker>(new PoseTracker(mPoseModel, 0));

            // Net
            this->spNets.emplace_back(
                std::make_shared<NetCaffe>(
                    this->mModelFolder + "pose/body_25b_video3/pose_deploy.prototxt",
                    this->mModelFolder + "pose/body_25b_video3/pose_iter_XXXXXX.caffemodel",
                    this->mGpuId, this->mEnableGoogleLogging));
//            this->spNets.emplace_back(
//                std::make_shared<NetCaffe>(
//                    this->mModelFolder + "pose/body_25b_video2/pose_deploy.prototxt",
//                    this->mModelFolder + "pose/body_25b_video2/pose_iter_2000.caffemodel",
//                    this->mGpuId, this->mEnableGoogleLogging));

            // Initialize
            this->spNets.back()->initializationOnThread();

            // Output in OP Format
            this->spCaffeNetOutputBlobs.emplace_back((this->spNets.back().get())->getOutputBlobArray());

            // Add curr outputs as reference
            this->mCurrPafBlobs.emplace_back((this->spNets.back().get())->getBlobArray("Mconv7_stage1_L2"));
            this->mCurrHmBlobs.emplace_back((this->spNets.back().get())->getBlobArray("Mconv7_stage2_L1"));
            this->mCurrTafBlobs.emplace_back((this->spNets.back().get())->getBlobArray("Mconv7_stage3_L4"));
            this->mCurrFmBlobs.emplace_back((this->spNets.back().get())->getBlobArray("conv4_4_CPM"));
            // Add last outputs as reference
            this->mLastPafBlobs.emplace_back((this->spNets.back().get())->getBlobArray("last_paf"));
            this->mLastHmBlobs.emplace_back((this->spNets.back().get())->getBlobArray("last_hm"));
            this->mLastTafBlobs.emplace_back((this->spNets.back().get())->getBlobArray("last_taf"));
            this->mLastFmBlobs.emplace_back((this->spNets.back().get())->getBlobArray("last_fm"));

        }
        else if(mPoseModel == PoseModel::BODY_21A)
        {
            // Tracker
            this->mPoseTracker = std::unique_ptr<PoseTracker>(new PoseTracker(mPoseModel, 1));

            // Net
            this->spNets.emplace_back(
                std::make_shared<NetCaffe>(
                    this->mModelFolder + "pose/body_21a_video/pose_deploy.prototxt",
                    this->mModelFolder + "pose/body_21a_video/pose_iter_264000.caffemodel",
                    this->mGpuId, this->mEnableGoogleLogging));

            // Initialize
            this->spNets.back()->initializationOnThread();

            // Output in OP Format
            this->spCaffeNetOutputBlobs.emplace_back((this->spNets.back().get())->getOutputBlobArray());

            // Add curr outputs as reference
            this->mCurrPafBlobs.emplace_back((this->spNets.back().get())->getBlobArray("Mconv7_stage3_L2_cont2"));
            this->mCurrHmBlobs.emplace_back((this->spNets.back().get())->getBlobArray("Mconv7_stage4_L1_cont2"));
            this->mCurrTafBlobs.emplace_back((this->spNets.back().get())->getBlobArray("output_taf"));
            this->mCurrFmBlobs.emplace_back((this->spNets.back().get())->getBlobArray("conv4_4_CPM"));
            // Add last outputs as reference
            this->mLastPafBlobs.emplace_back((this->spNets.back().get())->getBlobArray("last_paf"));
            this->mLastHmBlobs.emplace_back((this->spNets.back().get())->getBlobArray("last_hm"));
            //this->mLastTafBlobs.emplace_back((this->spNets.back().get())->getBlobArray("last_taf"));
            this->mLastFmBlobs.emplace_back((this->spNets.back().get())->getBlobArray("last_fm"));
        }
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
            spTafsBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};
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
        std::vector<std::shared_ptr<ArrayCpuGpu<float>>>& caffeTafBlobsShared,
        std::shared_ptr<ArrayCpuGpu<float>>& heatMapsBlob, std::shared_ptr<ArrayCpuGpu<float>>& peaksBlob,
        std::shared_ptr<ArrayCpuGpu<float>>& tafsBlob,
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

            // TAF Resize
            if(caffeTafBlobsShared.size())
            {
                float netFactor = netDescreaseFactor;
                float scaleFactor = 1.f/scaleInputToNetInput;
                const auto tafOutputBlobs = arraySharedToPtr2(caffeTafBlobsShared);
                auto topShape = tafOutputBlobs.at(0)->shape();
                topShape[2] = (int)std::round((topShape[2]*netFactor - 1.f) * scaleFactor) + 1;
                topShape[3] = (int)std::round((topShape[3]*netFactor - 1.f) * scaleFactor) + 1;
                tafsBlob->Reshape(topShape);
            }

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

            // Iterations
            int iterations = 1;

            // Reshape
            for (auto i = 0u ; i < inputNetData.size(); i++)
            {
                const auto changedVectors = !vectorsAreEqual(
                    mNetInput4DSizes.at(i), inputNetData[i].getSize());

                if (changedVectors)
                {
                    iterations = 2;

                    mNetInput4DSizes.at(i) = inputNetData[i].getSize();

                    // First reshape net
                    auto inputSize = inputNetData[i].getSize();
                    spNets.at(i)->reshape(inputNetData[i].getSize(), "image", 0);
                    spNets.at(i)->reshape(reduceSize(spNets.at(i)->shape("last_paf"), inputSize), "last_paf", 0);
                    spNets.at(i)->reshape(reduceSize(spNets.at(i)->shape("last_hm"), inputSize), "last_hm", 0);
                    if((this->mLastTafBlobs.size())) spNets.at(i)->reshape(reduceSize(spNets.at(i)->shape("last_taf"), inputSize), "last_taf", 0);
                    spNets.at(i)->reshape(reduceSize(spNets.at(i)->shape("last_fm"), inputSize), "last_fm", 1);

                    // Reshape Other
                    reshapePoseExtractorCaffe(
                        spResizeAndMergeCaffe, spNmsCaffe, spBodyPartConnectorCaffe,
                        spMaximumCaffe, spCaffeNetOutputBlobs, mCurrTafBlobs,
                        spHeatMapsBlob, spPeaksBlob, spTafsBlob, spMaximumPeaksBlob, 1.f, mPoseModel,
                        mGpuId, mUpsamplingRatio);

                    // Output Size
                    const auto ratio = (
                        mUpsamplingRatio <= 0.f
                            ? 1 : mUpsamplingRatio / getPoseNetDecreaseFactor(mPoseModel));
                    mNetOutputSize = Point<int>{
                        positiveIntRound(ratio*mNetInput4DSizes[0][3]),
                        positiveIntRound(ratio*mNetInput4DSizes[0][2])};
                }
            }

            // Forward Net
            for(int j = 0; j<iterations; j++){
                for (auto i = 0u ; i < inputNetData.size(); i++)
                {
                    spNets.at(i)->forwardPass(inputNetData[i]);

                    // dst, src
                    gpu_copy(this->mLastHmBlobs.at(i), this->mCurrHmBlobs.at(i));
                    gpu_copy(this->mLastPafBlobs.at(i), this->mCurrPafBlobs.at(i));
                    if((this->mLastTafBlobs.size())) gpu_copy(this->mLastTafBlobs.at(i), this->mCurrTafBlobs.at(i));
                    gpu_copy(this->mLastFmBlobs.at(i), this->mCurrFmBlobs.at(i));
                }
            }
            
            // Processing
            // 2. Resize heat maps + merge different scales
            // ~5ms (GPU) / ~20ms (CPU)
            const auto caffeNetOutputBlobs = arraySharedToPtr2(spCaffeNetOutputBlobs);
            const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
            spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
            spResizeAndMergeCaffe->Forward(caffeNetOutputBlobs, {spHeatMapsBlob.get()});
            // Get scale net to output (i.e., image input)
            // Note: In order to resize to input size, (un)comment the following lines
            const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
            const Point<int> netSize{
                positiveIntRound(scaleProducerToNetInput*inputDataSize.x),
                positiveIntRound(scaleProducerToNetInput*inputDataSize.y)};
            mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, inputDataSize)};
            // mScaleNetToOutput = 1.f;
            // 3. Get peaks by Non-Maximum Suppression
            // ~2ms (GPU) / ~7ms (CPU)
            const auto nmsThreshold = (float)get(PoseProperty::NMSThreshold);
            spNmsCaffe->setThreshold(nmsThreshold);
            const auto nmsOffset = float(0.5/double(mScaleNetToOutput));
            spNmsCaffe->setOffset(Point<float>{nmsOffset, nmsOffset});
            spNmsCaffe->Forward({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
            // 4. Connecting body parts
            spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
            spBodyPartConnectorCaffe->setInterMinAboveThreshold(
                (float)get(PoseProperty::ConnectInterMinAboveThreshold));
            spBodyPartConnectorCaffe->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
            spBodyPartConnectorCaffe->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
            spBodyPartConnectorCaffe->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));
            // Note: BODY_25D will crash (only implemented for CPU version)
            spBodyPartConnectorCaffe->Forward(
                {spHeatMapsBlob.get(), spPeaksBlob.get()}, mPoseKeypoints, mPoseScores);

            // Resize TAF
            const auto caffeTafBlobs = arraySharedToPtr2(mCurrTafBlobs);
            spResizeAndMergeCaffe->Forward(caffeTafBlobs, {spTafsBlob.get()});

            //return;

            // Run Tracker
            mPoseTracker->run(mPoseKeypoints, spTafsBlob, 1./mScaleNetToOutput);
            mPoseIds = mPoseTracker->getPoseIds();
            mPoseKeypoints = mPoseTracker->getPoseKeypoints();

//            cv::Mat mx = mat_from_blob(spTafsBlob, 0);
//            mx = cv::abs(mx);
//            cv::Mat my = mat_from_blob(spTafsBlob, 1);
//            my = cv::abs(my);
//            cv::Mat m = mx+my;
//            cv::imshow("win", m);
//            cv::waitKey(15);

            //std::cout << FLAGS_image_dir << std::endl;


            return;





//            // Set IDS
//            std::vector<long long> ids;
//            for (auto& kv : mPoseTracker->mTra) {
//                ids.emplace_back(kv.first);
//            }
//            mPoseIds.reset(ids.size());
//            for(int i=0; i<ids.size(); i++){
//                mPoseIds.at(i) = ids[i];
//            }

//            // Set Poses
//            op::Array<float> tracklet_keypoints({(int)upImpl->tracker.tracklets_internal.size(), 21, 3},0.0f);
//            int i=0;
//            for (auto& kv : upImpl->tracker.tracklets_internal) {
//                for(int j=0; j<mPoseKeypoints.getSize(1); j++)
//                    for(int k=0; k<mPoseKeypoints.getSize(2); k++)
//                        tracklet_keypoints.at({i,j,k}) = kv.second.kp.at({j,k});
//                i+=1;
//            }
//            mPoseKeypoints = tracklet_keypoints.clone();
//            // Scale Up
//            for(int i=0; i<mPoseKeypoints.getSize()[0]; i++){
//                op::Array<float> person_kp = get_person_no_copy(mPoseKeypoints, i);
//                rescale_kp(person_kp, mScaleNetToOutput);
//            }




            //std::cout << mCurrHmBlobs.at(0)->shape_string() << std::endl;
            //std::cout << spHeatMapsBlob->shape_string() << std::endl;


            //std::cout << floatScaleRatios[0] << std::endl;
//            spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
//            spResizeAndMergeCaffe->Forward(caffeNetOutputBlobs, {spHeatMapsBlob.get()});

//            // Tracking
//            mPoseIds.reset(mPoseKeypoints.getSize(0));
//            for(int i=0; i<mPoseIds.getSize(0); i++) mPoseIds.at(i) = i;

            //std::cout << mPoseIds.getSize(0) << std::endl;

        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}

// openpose_staf
// build/examples/openpose/openpose.bin --model_pose BODY_25B --tracking 1  --render_pose 1 --disable_multi_thread
// build/examples/openpose/openpose.bin --model_pose BODY_25B --tracking 1 --image_dir eval/posetrack/images/val/015302_mpii_test --render_pose 1 --disable_multi_thread
// build/examples/openpose/openpose.bin --model_pose BODY_25B --tracking 1 --image_dir eval/posetrack/images/val/015302_mpii_test --render_pose 1

// openpose_oldtracking
// build/examples/openpose/openpose.bin --model_pose BODY_21A --render_pose 1 --image_dir eval/posetrack/images/val/015302_mpii_test

// python d_setLayers_raaj.py video 2 1
