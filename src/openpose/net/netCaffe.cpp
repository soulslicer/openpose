#include <numeric> // std::accumulate

#include <atomic>
#include <mutex>
#include <glog/logging.h> // google::InitGoogleLogging

#ifdef USE_PYTORCH
    #include <torch/torch.h>
    #include <torch/script.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
#elif USE_CAFFE
    #include <caffe/net.hpp>
#endif

#ifdef USE_CUDA
    #include <openpose/gpu/cuda.hpp>
#endif
#ifdef USE_OPENCL
    #include <openpose/gpu/opencl.hcl>
    #include <openpose/gpu/cl2.hpp>
#endif
#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/net/netCaffe.hpp>

// HACK
//#include <caffe/net.hpp>

#include <iostream>

namespace op
{
    std::mutex sMutexNetCaffe;
    std::atomic<bool> sGoogleLoggingInitialized{false};
    #ifdef USE_OPENCL
        std::atomic<bool> sOpenCLInitialized{false};
    #endif

    struct NetCaffe::ImplNetCaffe
    {        
        // Init with constructor
        const int mGpuId;
        const std::string mCaffeProto;
        const std::string mCaffeTrainedModel;
        const std::string mLastBlobName;
        std::vector<int> mNetInputSize4D;
        // Init with thread
        #ifdef USE_PYTORCH
            std::shared_ptr<torch::jit::script::Module> upTorchNet;
            std::shared_ptr<torch::Tensor> spInputBlob;
            std::shared_ptr<torch::Tensor> spOutputBlob;
        #elif NV_CAFFE
            std::unique_ptr<caffe::Net> upCaffeNet;
            boost::shared_ptr<caffe::TBlob<float>> spOutputBlob;
        #else
            std::unique_ptr<caffe::Net<float>> upCaffeNet;
            boost::shared_ptr<caffe::Blob<float>> spOutputBlob;
        #endif

        ImplNetCaffe(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId,
                     const bool enableGoogleLogging, const std::string& lastBlobName) :
            mGpuId{gpuId},
            mCaffeProto{caffeProto},
            mCaffeTrainedModel{caffeTrainedModel},
            mLastBlobName{lastBlobName}
        {
            try
            {
                const std::string message{".\nPossible causes:\n\t1. Not downloading the OpenPose trained models."
                                          "\n\t2. Not running OpenPose from the same directory where the `model`"
                                          " folder is located.\n\t3. Using paths with spaces."};
                if (!existFile(mCaffeProto))
                    error("Prototxt file not found: " + mCaffeProto + message, __LINE__, __FUNCTION__, __FILE__);
                if (!existFile(mCaffeTrainedModel))
                    error("Caffe trained model file not found: " + mCaffeTrainedModel + message,
                          __LINE__, __FUNCTION__, __FILE__);
                // Double if condition in order to speed up the program if it is called several times
                if (enableGoogleLogging && !sGoogleLoggingInitialized)
                {
                    std::lock_guard<std::mutex> lock{sMutexNetCaffe};
                    if (enableGoogleLogging && !sGoogleLoggingInitialized)
                    {
                        google::InitGoogleLogging("OpenPose");
                        sGoogleLoggingInitialized = true;
                    }
                }
                #ifdef USE_OPENCL
                    // Initialize OpenCL
                    if (!sOpenCLInitialized)
                    {
                        std::lock_guard<std::mutex> lock{sMutexNetCaffe};
                        if (!sOpenCLInitialized)
                        {
                            caffe::Caffe::set_mode(caffe::Caffe::GPU);
                            std::vector<int> devices;
                            const int maxNumberGpu = OpenCL::getTotalGPU();
                            for (auto i = 0; i < maxNumberGpu; i++)
                                devices.emplace_back(i);
                            caffe::Caffe::SetDevices(devices);
                            if (mGpuId >= maxNumberGpu)
                                error("Unexpected error. Please, notify us.", __LINE__, __FUNCTION__, __FILE__);
                            sOpenCLInitialized = true;
                        }
                    }
                #endif
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    };

    #ifdef USE_PYTORCH
        inline void reshapeNetTorch(torch::jit::script::Module* caffeNet, const std::vector<int>& dimensions)
        {
            try
            {
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    #elif USE_CAFFE
        #ifdef NV_CAFFE
        inline void reshapeNetCaffe(caffe::Net* caffeNet, const std::vector<int>& dimensions)
        #else
        inline void reshapeNetCaffe(caffe::Net<float>* caffeNet, const std::vector<int>& dimensions)
        #endif
        {
            try
            {
                caffeNet->blobs()[0]->Reshape(dimensions);
                caffeNet->Reshape();
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

    NetCaffe::NetCaffe(const std::string& caffeProto, const std::string& caffeTrainedModel, const int gpuId,
                       const bool enableGoogleLogging, const std::string& lastBlobName)
        #ifdef USE_CAFFE
            : upImpl{new ImplNetCaffe{caffeProto, caffeTrainedModel, gpuId, enableGoogleLogging,
                                      lastBlobName}}
        #endif
    {
        try
        {
            #ifndef USE_CAFFE
                UNUSED(caffeProto);
                UNUSED(caffeTrainedModel);
                UNUSED(gpuId);
                UNUSED(enableGoogleLogging);
                UNUSED(lastBlobName);
                error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    NetCaffe::~NetCaffe()
    {
    }

    void NetCaffe::initializationOnThread()
    {
        try
        {
            #ifdef USE_PYTORCH
                upImpl->upTorchNet = torch::jit::load(upImpl->mCaffeTrainedModel);
                upImpl->spInputBlob.reset(new torch::Tensor{torch::zeros({1,1,1,1}).cuda()});
                //Array<float> inputData({1,3,656,368});
                //this->forwardPass(inputData);
            #elif USE_CAFFE
                // Initialize net
                #ifdef USE_OPENCL
                    caffe::Caffe::set_mode(caffe::Caffe::GPU);
                    caffe::Caffe::SelectDevice(upImpl->mGpuId, true);
                    upImpl->upCaffeNet.reset(new caffe::Net<float>{upImpl->mCaffeProto, caffe::TEST,
                                             caffe::Caffe::GetDefaultDevice()});
                    upImpl->upCaffeNet->CopyTrainedLayersFrom(upImpl->mCaffeTrainedModel);
                    OpenCL::getInstance(upImpl->mGpuId, CL_DEVICE_TYPE_GPU, true);
                #else
                    #ifdef USE_CUDA
                        caffe::Caffe::set_mode(caffe::Caffe::GPU);
                        caffe::Caffe::SetDevice(upImpl->mGpuId);
                        #ifdef NV_CAFFE
                            upImpl->upCaffeNet.reset(new caffe::Net{upImpl->mCaffeProto, caffe::TEST});
                        #else
                            upImpl->upCaffeNet.reset(new caffe::Net<float>{upImpl->mCaffeProto, caffe::TEST});
                        #endif
                    #else
                        caffe::Caffe::set_mode(caffe::Caffe::CPU);
                        #ifdef _WIN32
                            upImpl->upCaffeNet.reset(new caffe::Net<float>{upImpl->mCaffeProto, caffe::TEST,
                                                                           caffe::Caffe::GetCPUDevice()});
                        #else
                            upImpl->upCaffeNet.reset(new caffe::Net<float>{upImpl->mCaffeProto, caffe::TEST});
                        #endif
                    #endif
                    upImpl->upCaffeNet->CopyTrainedLayersFrom(upImpl->mCaffeTrainedModel);
                    #ifdef USE_CUDA
                        cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                    #endif
                #endif
                // Set spOutputBlob
                #ifdef NV_CAFFE
                    upImpl->spOutputBlob = boost::static_pointer_cast<caffe::TBlob<float>>(
                        upImpl->upCaffeNet->blob_by_name(upImpl->mLastBlobName));
                #else
                    upImpl->spOutputBlob = upImpl->upCaffeNet->blob_by_name(upImpl->mLastBlobName);
                #endif
                // Sanity check
                if (upImpl->spOutputBlob == nullptr)
                    error("The output blob is a nullptr. Did you use the same name than the prototxt? (Used: "
                          + upImpl->mLastBlobName + ").", __LINE__, __FUNCTION__, __FILE__);
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void NetCaffe::forwardPass(const Array<float>& inputData) const
    {
        try
        {
            #ifdef USE_PYTORCH
                // Sanity checks
                if (inputData.empty())
                    error("The Array inputData cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
                if (inputData.getNumberDimensions() != 4 || inputData.getSize(1) != 3)
                    error("The Array inputData must have 4 dimensions: [batch size, 3 (RGB), height, width].",
                          __LINE__, __FUNCTION__, __FILE__);

                // Set Input
                upImpl->spInputBlob->resize_(inputData.getSizeLong());
                cudaMemcpy(upImpl->spInputBlob->data_ptr(), inputData.getConstPtr(), inputData.getVolume() * sizeof(float),
                           cudaMemcpyHostToDevice);

                // Create a vector of inputs.
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(*upImpl->spInputBlob.get());

                // Forward
                torch::Tensor output = upImpl->upTorchNet->forward(inputs).toTensor();

                // Inefficient this makes a copy (What to do? Use the trick in arrayCpuGpu)
                upImpl->spOutputBlob = std::make_shared<torch::Tensor>(output);

            #elif USE_CAFFE
                // Sanity checks
                if (inputData.empty())
                    error("The Array inputData cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
                if (inputData.getNumberDimensions() != 4 || inputData.getSize(1) != 3)
                    error("The Array inputData must have 4 dimensions: [batch size, 3 (RGB), height, width].",
                          __LINE__, __FUNCTION__, __FILE__);
                // Reshape Caffe net if required
                if (!vectorsAreEqual(upImpl->mNetInputSize4D, inputData.getSize()))
                {
                    upImpl->mNetInputSize4D = inputData.getSize();
                    reshapeNetCaffe(upImpl->upCaffeNet.get(), inputData.getSize());
                }
                // Copy frame data to GPU memory
                #ifdef USE_CUDA
                    #ifdef NV_CAFFE
                        auto* gpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_gpu_data<float>();
                    #else
                        auto* gpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_gpu_data();
                    #endif
                    cudaMemcpy(gpuImagePtr, inputData.getConstPtr(), inputData.getVolume() * sizeof(float),
                               cudaMemcpyHostToDevice);
                #elif defined USE_OPENCL
                    auto* gpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_gpu_data();
                    cl::Buffer imageBuffer = cl::Buffer((cl_mem)gpuImagePtr, true);
                    OpenCL::getInstance(upImpl->mGpuId)->getQueue().enqueueWriteBuffer(
                        imageBuffer, true, 0, inputData.getVolume() * sizeof(float), inputData.getConstPtr());
                #else
                    auto* cpuImagePtr = upImpl->upCaffeNet->blobs().at(0)->mutable_cpu_data();
                    std::copy(inputData.getConstPtr(), inputData.getConstPtr() + inputData.getVolume(), cpuImagePtr);
                #endif
                // Perform deep network forward pass
                upImpl->upCaffeNet->ForwardFrom(0);
                // Cuda checks
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            #else
                UNUSED(inputData);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    std::shared_ptr<ArrayCpuGpu<float>> NetCaffe::getOutputBlobArray() const
    {
        try
        {
            #ifdef USE_PYTORCH
                return std::make_shared<ArrayCpuGpu<float>>(upImpl->spOutputBlob.get());
            #elif USE_CAFFE
                return std::make_shared<ArrayCpuGpu<float>>(upImpl->spOutputBlob.get());
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
