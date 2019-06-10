#include <limits> // std::numeric_limits
#include <openpose/gpu/cuda.hpp>
#ifdef USE_CUDA
    #include <openpose/gpu/cuda.hu>
#endif
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/pose/poseExtractorCaffeTaf.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace op
{
    const bool TOP_DOWN_REFINEMENT = false; // Note: +5% acc 1 scale, -2% max acc setting

    void gpu_copy2(std::shared_ptr<ArrayCpuGpu<float>> dst, std::shared_ptr<ArrayCpuGpu<float>> src)
    {
        size_t size = sizeof(float)*src->shape()[1]*src->shape()[2]*src->shape()[3];
        cudaMemcpy(dst->mutable_gpu_data(), src->gpu_data(), size, cudaMemcpyDeviceToDevice);
    }

    cv::Mat mat_from_blob2(std::shared_ptr<ArrayCpuGpu<float>> src, int channel)
    {
        cv::Mat m(src->shape()[2], src->shape()[3], CV_32FC1, (float*)(src->cpu_data() + channel*src->shape()[2]*src->shape()[3]));
        return m;
    }


    #ifdef USE_CAFFE
        std::vector<ArrayCpuGpu<float>*> arraySharedToPtr3(
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
                const auto caffeNetOutputBlobs = arraySharedToPtr3(caffeNetOutputBlobsShared);
                resizeAndMergeCaffe->Reshape(
                    caffeNetOutputBlobs, {heatMapsBlob.get()},
                    netDescreaseFactor, 1.f/scaleInputToNetInput, true, gpuId);
                // Pose extractor blob and layer
                nmsCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()}, getPoseMaxPeaks(),
                                  getPoseNumberBodyParts(poseModel), gpuId);
                // Pose extractor blob and layer
                bodyPartConnectorCaffe->Reshape({heatMapsBlob.get(), peaksBlob.get()}, gpuId);
                if (TOP_DOWN_REFINEMENT)
                    maximumCaffe->Reshape({heatMapsBlob.get()}, {maximumPeaksBlob.get()});

                // TAF Resize
                if(caffeTafBlobsShared.size())
                {
                    float netFactor = netDescreaseFactor;
                    float scaleFactor = 1.f/scaleInputToNetInput;
                    const auto tafOutputBlobs = arraySharedToPtr3(caffeTafBlobsShared);
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

        void PoseExtractorCaffeTaf::addCaffeNetOnThread(
            std::vector<std::shared_ptr<Net>>& net,
            std::vector<std::shared_ptr<ArrayCpuGpu<float>>>& caffeNetOutputBlob,
            const PoseModel poseModel, const int gpuId, const std::string& modelFolder,
            const std::string& protoTxtPath, const std::string& caffeModelPath, const bool enableGoogleLogging)
        {
            try
            {
                // Tracker
                this->mPoseTracker = std::unique_ptr<PoseTracker>(new PoseTracker(mPoseModel, 2));

                // Add Caffe Net
                net.emplace_back(
                    std::make_shared<NetCaffe>(
                        modelFolder + (protoTxtPath.empty() ? getPoseProtoTxt(poseModel) : protoTxtPath),
                        modelFolder + (caffeModelPath.empty() ? getPoseTrainedModel(poseModel) : caffeModelPath),
                        gpuId, enableGoogleLogging));
                // net.emplace_back(
                //     std::make_shared<NetOpenCv>(
                //         modelFolder + (protoTxtPath.empty() ? getPoseProtoTxt(poseModel) : protoTxtPath),
                //         modelFolder + (caffeModelPath.empty() ? getPoseTrainedModel(poseModel) : caffeModelPath),
                //         gpuId));
                // UNUSED(enableGoogleLogging);
                // Initializing them on the thread
                net.back()->initializationOnThread();
                caffeNetOutputBlob.emplace_back((net.back().get())->getOutputBlobArray());
                // Sanity check
                if (net.size() != caffeNetOutputBlob.size())
                    error("Weird error, this should not happen. Notify us.", __LINE__, __FUNCTION__, __FILE__);

                // TAF
                this->spNetsTrack.emplace_back(
                    std::make_shared<NetCaffe>(
                        this->mModelFolder + "pose/body_25b_video4/pose_deploy.prototxt",
                        this->mModelFolder + "pose/body_25b_video4/pose_iter_XXXXXX.caffemodel",
                        this->mGpuId, this->mEnableGoogleLogging, "Mconv7_stage0_L4"));

                this->spNetsTrack.back()->initializationOnThread();

                // Add curr outputs as reference
                this->mCurrPafBlobs.emplace_back((this->spNets.back().get())->getBlobArray("Mconv7_stage1_L2"));
                this->mCurrFmBlobs.emplace_back((this->spNets.back().get())->getBlobArray("conv4_4_CPM"));
                // Add last outputs as reference
                this->mTrackCurrTafBlobs.emplace_back((this->spNetsTrack.back().get())->getBlobArray("Mconv7_stage0_L4"));
                this->mTrackLastTafBlobs.emplace_back((this->spNetsTrack.back().get())->getBlobArray("last_taf"));
                this->mTrackCurrFmBlobs.emplace_back((this->spNetsTrack.back().get())->getBlobArray("curr_fm"));
                this->mTrackLastFmBlobs.emplace_back((this->spNetsTrack.back().get())->getBlobArray("last_fm"));
                this->mTrackCurrPafBlobs.emplace_back((this->spNetsTrack.back().get())->getBlobArray("curr_paf"));
                this->mTrackLastPafBlobs.emplace_back((this->spNetsTrack.back().get())->getBlobArray("last_paf"));

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
    #endif

    PoseExtractorCaffeTaf::PoseExtractorCaffeTaf(
        const PoseModel poseModel, const std::string& modelFolder, const int gpuId,
        const std::vector<HeatMapType>& heatMapTypes, const ScaleMode heatMapScaleMode, const bool addPartCandidates,
        const bool maximizePositives, const std::string& protoTxtPath, const std::string& caffeModelPath,
        const float upsamplingRatio, const bool enableNet, const bool enableGoogleLogging) :
        PoseExtractorNet{poseModel, heatMapTypes, heatMapScaleMode, addPartCandidates, maximizePositives},
        mPoseModel{poseModel},
        mGpuId{gpuId},
        mModelFolder{modelFolder},
        mProtoTxtPath{protoTxtPath},
        mCaffeModelPath{caffeModelPath},
        mUpsamplingRatio{upsamplingRatio},
        mEnableNet{enableNet},
        mEnableGoogleLogging{enableGoogleLogging}
        #ifdef USE_CAFFE
            ,
            spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
            spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
            spBodyPartConnectorCaffe{std::make_shared<BodyPartConnectorCaffe<float>>()},
            spMaximumCaffe{(TOP_DOWN_REFINEMENT ? std::make_shared<MaximumCaffe<float>>() : nullptr)}
        #endif
    {
        try
        {
            #ifdef USE_CAFFE
                // Layers parameters
                spBodyPartConnectorCaffe->setPoseModel(mPoseModel);
                spBodyPartConnectorCaffe->setMaximizePositives(maximizePositives);
            #else
                UNUSED(poseModel);
                UNUSED(modelFolder);
                UNUSED(gpuId);
                UNUSED(heatMapTypes);
                UNUSED(heatMapScaleMode);
                UNUSED(addPartCandidates);
                UNUSED(maximizePositives);
                UNUSED(protoTxtPath);
                UNUSED(caffeModelPath);
                UNUSED(enableGoogleLogging);
                error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PoseExtractorCaffeTaf::~PoseExtractorCaffeTaf()
    {
    }

    void PoseExtractorCaffeTaf::netInitializationOnThread()
    {
        try
        {
            #ifdef USE_CAFFE
                if (mEnableNet)
                {
                    // Logging
                    log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    // Initialize Caffe net
                    addCaffeNetOnThread(
                        spNets, spCaffeNetOutputBlobs, mPoseModel, mGpuId,
                        mModelFolder, mProtoTxtPath, mCaffeModelPath,
                        mEnableGoogleLogging);
                    #ifdef USE_CUDA
                        cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                    #endif
                }
                // Initialize blobs
                spHeatMapsBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};
                spPeaksBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};
                spTafsBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};

                if (TOP_DOWN_REFINEMENT)
                    spMaximumPeaksBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
                // Logging
                log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::vector<int> reduceSize2(const std::vector<int>& outputSize, const std::vector<int>& inputSize, int stride=8)
    {
        std::vector<int> finalSize(outputSize);
        finalSize[2] = (inputSize[2]/stride);
        finalSize[3] = (inputSize[3]/stride);
        return finalSize;
    }

    void PoseExtractorCaffeTaf::forwardPass(
        const std::vector<Array<float>>& inputNetData, const Point<int>& inputDataSize,
        const std::vector<double>& scaleInputToNetInputs, const Array<float>& poseNetOutput)
    {
        try
        {
            #ifdef USE_CAFFE
                // const auto REPS = 1;
                // double timeNormalize1 = 0.;
                // double timeNormalize2 = 0.;
                // double timeNormalize3 = 0.;
                // double timeNormalize4 = 0.;
                // OP_CUDA_PROFILE_INIT(REPS);
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

                // Process each image - Caffe deep network
                if (mEnableNet)
                {
                    while (spNets.size() < numberScales)
                        addCaffeNetOnThread(
                            spNets, spCaffeNetOutputBlobs, mPoseModel, mGpuId,
                            mModelFolder, mProtoTxtPath, mCaffeModelPath, false);

                    for (auto i = 0u ; i < inputNetData.size(); i++)
                        spNets.at(i)->forwardPass(inputNetData[i]);
                }
                // If custom network output
                else
                {
                    // Sanity check
                    if (inputNetData.size() != 1u)
                        error("Size(inputNetData) must match the provided heatmaps batch size ("
                              + std::to_string(inputNetData.size()) + " vs. " + std::to_string(1) + ").",
                              __LINE__, __FUNCTION__, __FILE__);
                    // Copy heatmap information
                    spCaffeNetOutputBlobs.clear();
                    const bool copyFromGpu = false;
                    spCaffeNetOutputBlobs.emplace_back(
                        std::make_shared<ArrayCpuGpu<float>>(poseNetOutput, copyFromGpu));
                }
                // Reshape blobs if required
                for (auto i = 0u ; i < inputNetData.size(); i++)
                {
                    // Reshape blobs if required - For dynamic sizes (e.g., images of different aspect ratio)
                    const auto changedVectors = !vectorsAreEqual(
                        mNetInput4DSizes.at(i), inputNetData[i].getSize());
                    if (changedVectors)
                    {
                        mNetInput4DSizes.at(i) = inputNetData[i].getSize();
                        reshapePoseExtractorCaffe(
                            spResizeAndMergeCaffe, spNmsCaffe, spBodyPartConnectorCaffe,
                            spMaximumCaffe, spCaffeNetOutputBlobs, mTrackCurrTafBlobs, spHeatMapsBlob,
                            spPeaksBlob, spTafsBlob, spMaximumPeaksBlob, 1.f, mPoseModel,
                            mGpuId, mUpsamplingRatio);
                            // In order to resize to input size to have same results as Matlab
                            // scaleInputToNetInputs[i] vs. 1.f

                        // Tracking stuff here
                        auto inputSize = inputNetData[i].getSize();
                        spNetsTrack.at(i)->reshape(reduceSize2(spNetsTrack.at(i)->shape("last_taf"), inputSize), "last_taf", 0);
                        spNetsTrack.at(i)->reshape(reduceSize2(spNetsTrack.at(i)->shape("curr_fm"), inputSize), "curr_fm", 0);
                        spNetsTrack.at(i)->reshape(reduceSize2(spNetsTrack.at(i)->shape("last_fm"), inputSize), "last_fm", 0);
                        spNetsTrack.at(i)->reshape(reduceSize2(spNetsTrack.at(i)->shape("last_paf"), inputSize), "last_paf", 0);
                        spNetsTrack.at(i)->reshape(reduceSize2(spNetsTrack.at(i)->shape("curr_paf"), inputSize), "curr_paf", 1);

//                        std::cout << inputSize.at(2) << " " << inputSize.at(3) << std::endl;
//                        std::cout << this->mTrackCurrTafBlobs.at(0)->shape_string() << std::endl;
//                        exit(-1);


                    }
                    // Get scale net to output (i.e., image input)
                    const auto ratio = (
                        mUpsamplingRatio <= 0.f
                            ? 1 : mUpsamplingRatio / getPoseNetDecreaseFactor(mPoseModel));
                    if (changedVectors || TOP_DOWN_REFINEMENT)
                        mNetOutputSize = Point<int>{
                            positiveIntRound(ratio*mNetInput4DSizes[0][3]),
                            positiveIntRound(ratio*mNetInput4DSizes[0][2])};
                }
                // OP_CUDA_PROFILE_END(timeNormalize1, 1e3, REPS);
                // OP_CUDA_PROFILE_INIT(REPS);
                // 2. Resize heat maps + merge different scales
                // ~5ms (GPU) / ~20ms (CPU)
                const auto caffeNetOutputBlobs = arraySharedToPtr3(spCaffeNetOutputBlobs);
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
                // OP_CUDA_PROFILE_END(timeNormalize2, 1e3, REPS);
                const auto nmsThreshold = (float)get(PoseProperty::NMSThreshold);
                const auto nmsOffset = float(0.5/double(mScaleNetToOutput));
                // OP_CUDA_PROFILE_INIT(REPS);
                spNmsCaffe->setThreshold(nmsThreshold);
                spNmsCaffe->setOffset(Point<float>{nmsOffset, nmsOffset});
                spNmsCaffe->Forward({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
                // 4. Connecting body parts
                // OP_CUDA_PROFILE_END(timeNormalize3, 1e3, REPS);
                // OP_CUDA_PROFILE_INIT(REPS);
                spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
                spBodyPartConnectorCaffe->setInterMinAboveThreshold(
                    (float)get(PoseProperty::ConnectInterMinAboveThreshold));
                spBodyPartConnectorCaffe->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
                spBodyPartConnectorCaffe->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
                spBodyPartConnectorCaffe->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));
                // Note: BODY_25D will crash (only implemented for CPU version)
                spBodyPartConnectorCaffe->Forward(
                    {spHeatMapsBlob.get(), spPeaksBlob.get()}, mPoseKeypoints, mPoseScores);
                // OP_CUDA_PROFILE_END(timeNormalize4, 1e3, REPS);
                // log("1(caf)= " + std::to_string(timeNormalize1) + "ms");
                // log("2(res) = " + std::to_string(timeNormalize2) + " ms");
                // log("3(nms) = " + std::to_string(timeNormalize3) + " ms");
                // log("4(bpp) = " + std::to_string(timeNormalize4) + " ms");
                // Re-run on each person

                // Tracking related
                int iterations = 1;
                for(int j = 0; j<iterations; j++){
                    for (auto i = 0u ; i < inputNetData.size(); i++)
                    {
                        gpu_copy2(this->mTrackLastTafBlobs.at(i), this->mTrackCurrTafBlobs.at(i));
                        gpu_copy2(this->mTrackLastFmBlobs.at(i), this->mTrackCurrFmBlobs.at(i));
                        gpu_copy2(this->mTrackCurrFmBlobs.at(i), this->mCurrFmBlobs.at(i));
                        gpu_copy2(this->mTrackLastPafBlobs.at(i), this->mTrackCurrPafBlobs.at(i));
                        gpu_copy2(this->mTrackCurrPafBlobs.at(i), this->mCurrPafBlobs.at(i));
                        spNetsTrack.at(i)->forwardPass();
                    }
                }

                const auto caffeTafBlobs = arraySharedToPtr3(mTrackCurrTafBlobs);
                spResizeAndMergeCaffe->Forward(caffeTafBlobs, {spTafsBlob.get()});

//                    cv::Mat mx = mat_from_blob2(spTafsBlob, 0);
//                    mx = cv::abs(mx);
//                    cv::Mat my = mat_from_blob2(spTafsBlob, 1);
//                    my = cv::abs(my);
//                    cv::Mat m = mx+my;
//                    cv::imshow("win", m);
//                    cv::waitKey(15);

                // 5. CUDA sanity check
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            #else
                UNUSED(inputNetData);
                UNUSED(inputDataSize);
                UNUSED(scaleInputToNetInputs);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    const float* PoseExtractorCaffeTaf::getCandidatesCpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return spPeaksBlob->cpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorCaffeTaf::getCandidatesGpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return spPeaksBlob->gpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorCaffeTaf::getHeatMapCpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return spHeatMapsBlob->cpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorCaffeTaf::getHeatMapGpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return spHeatMapsBlob->gpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    std::vector<int> PoseExtractorCaffeTaf::getHeatMapSize() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return spHeatMapsBlob->shape();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    const float* PoseExtractorCaffeTaf::getPoseGpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                error("GPU pointer for people pose data not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
                checkThread();
                return nullptr;
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
}
