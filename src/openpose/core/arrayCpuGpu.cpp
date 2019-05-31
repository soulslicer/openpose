#ifdef USE_PYTORCH
    #include <torch/torch.h>
#elif USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/core/arrayCpuGpu.hpp>

namespace op
{
    template<typename T>
    struct ArrayCpuGpu<T>::ImplArrayCpuGpu
    {
        #ifdef USE_PYTORCH
            std::unique_ptr<torch::Tensor> upTorchBlobT;
            torch::Tensor* pTorchBlobT;
        #elif USE_CAFFE
            #ifdef NV_CAFFE
                std::unique_ptr<caffe::TBlob<T>> upCaffeBlobT;
                caffe::TBlob<T>* pCaffeBlobT;
            #else
                std::unique_ptr<caffe::Blob<T>> upCaffeBlobT;
                caffe::Blob<T>* pCaffeBlobT;
            #endif
        #endif
    };

    const std::string constructorErrorMessage = "ArrayCpuGpu class only implemented for Caffe DL framework (enable"
        " `USE_CAFFE` in CMake-GUI).";
    template<typename T>
    ArrayCpuGpu<T>::ArrayCpuGpu()
    {
        try
        {
            #ifdef USE_PYTORCH
                // Construct spImpl
                spImpl.reset(new ImplArrayCpuGpu{});
                spImpl->upTorchBlobT.reset(new torch::Tensor{torch::zeros({1})});
                spImpl->pTorchBlobT = spImpl->upTorchBlobT.get();
            #elif USE_CAFFE
                // Construct spImpl
                spImpl.reset(new ImplArrayCpuGpu{});
                #ifdef NV_CAFFE
                    spImpl->upCaffeBlobT.reset(new caffe::TBlob<T>{});
                #else
                    spImpl->upCaffeBlobT.reset(new caffe::Blob<T>{});
                #endif
                spImpl->pCaffeBlobT = spImpl->upCaffeBlobT.get();
            #else
                error(constructorErrorMessage, __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    ArrayCpuGpu<T>::ArrayCpuGpu(const void* caffeBlobTPtr)
    {
        try
        {
            #ifdef USE_PYTORCH
                // Construct spImpl
                spImpl.reset(new ImplArrayCpuGpu{});
                spImpl->pTorchBlobT = (torch::Tensor*)caffeBlobTPtr;
            #elif USE_CAFFE
                // Construct spImpl
                spImpl.reset(new ImplArrayCpuGpu{});
                #ifdef NV_CAFFE
                    spImpl->pCaffeBlobT = (caffe::TBlob<T>*)caffeBlobTPtr;
                #else
                    spImpl->pCaffeBlobT = (caffe::Blob<T>*)caffeBlobTPtr;
                #endif
            #else
                UNUSED(caffeBlobTPtr);
                error(constructorErrorMessage, __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    ArrayCpuGpu<T>::ArrayCpuGpu(const Array<T>& array, const bool copyFromGpu)
    {
        try
        {
            #ifdef USE_PYTORCH
                // Get updated size
                std::vector<int> arraySize;
                // If batch size = 1 --> E.g., array.getSize() == {78, 368, 368}
                if (array.getNumberDimensions() == 3)
                    // Add 1: arraySize = {1}
                    arraySize.emplace_back(1);
                // Add {78, 368, 368}: arraySize = {1, 78, 368, 368}
                for (const auto& sizeI : array.getSize())
                    arraySize.emplace_back(sizeI);
                // Convert to Long
                std::vector<long> arraySizeLong(arraySize.begin(), arraySize.end());
                // Construct spImpl
                spImpl.reset(new ImplArrayCpuGpu{});
                spImpl->upTorchBlobT.reset(new torch::Tensor(torch::zeros({1})));
                spImpl->upTorchBlobT->resize_(arraySizeLong);
                spImpl->pTorchBlobT = spImpl->upTorchBlobT.get();
                // Copy data
                // CPU copy
                if (!copyFromGpu)
                {
                    const auto* const arrayPtr = array.getConstPtr();
                    std::copy(arrayPtr, arrayPtr + array.getVolume(), (T*)spImpl->pTorchBlobT->cpu().data_ptr());
                }
                // GPU copy
                else
                    error("Not implemented yet. Let us know you are interested on this function.",
                          __LINE__, __FUNCTION__, __FILE__);
            #elif USE_CAFFE
                // Get updated size
                std::vector<int> arraySize;
                // If batch size = 1 --> E.g., array.getSize() == {78, 368, 368}
                if (array.getNumberDimensions() == 3)
                    // Add 1: arraySize = {1}
                    arraySize.emplace_back(1);
                // Add {78, 368, 368}: arraySize = {1, 78, 368, 368}
                for (const auto& sizeI : array.getSize())
                    arraySize.emplace_back(sizeI);
                // Construct spImpl
                spImpl.reset(new ImplArrayCpuGpu{});
                #ifdef NV_CAFFE
                    spImpl->upCaffeBlobT.reset(new caffe::TBlob<T>{arraySize});
                #else
                    spImpl->upCaffeBlobT.reset(new caffe::Blob<T>{arraySize});
                #endif
                spImpl->pCaffeBlobT = spImpl->upCaffeBlobT.get();
                // Copy data
                // CPU copy
                if (!copyFromGpu)
                {
                    const auto* const arrayPtr = array.getConstPtr();
                    std::copy(arrayPtr, arrayPtr + array.getVolume(), spImpl->pCaffeBlobT->mutable_cpu_data());
                }
                // GPU copy
                else
                    error("Not implemented yet. Let us know you are interested on this function.",
                          __LINE__, __FUNCTION__, __FILE__);
            #else
                UNUSED(array);
                UNUSED(copyFromGpu);
                error(constructorErrorMessage, __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    ArrayCpuGpu<T>::ArrayCpuGpu(const int num, const int channels, const int height, const int width)
    {
        try
        {
            #ifdef USE_PYTORCH
                // Construct spImpl
                spImpl.reset(new ImplArrayCpuGpu{});
                spImpl->upTorchBlobT.reset(new torch::Tensor(torch::zeros({1})));
                spImpl->upTorchBlobT->resize_({num, channels, height, width});
                spImpl->pTorchBlobT = spImpl->upTorchBlobT.get();
            #elif USE_CAFFE
                // Construct spImpl
                spImpl.reset(new ImplArrayCpuGpu{});
                #ifdef NV_CAFFE
                    spImpl->upCaffeBlobT.reset(new caffe::TBlob<T>{num, channels, height, width});
                #else
                    spImpl->upCaffeBlobT.reset(new caffe::Blob<T>{num, channels, height, width});
                #endif
                spImpl->pCaffeBlobT = spImpl->upCaffeBlobT.get();
            #else
                UNUSED(num);
                UNUSED(channels);
                UNUSED(height);
                UNUSED(width);
                error(constructorErrorMessage, __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    // template<typename T>
    // ArrayCpuGpu<T>::ArrayCpuGpu(const std::vector<int>& shape)
    // {
    //     try
    //     {
    //         #ifdef USE_CAFFE
    //             spImpl.reset(new ImplArrayCpuGpu{});
    //             spImpl->upCaffeBlobT.reset(new caffe::Blob<T>{shape});
    //             spImpl->pCaffeBlobT = spImpl->upCaffeBlobT.get();
    //         #else
    //             error(constructorErrorMessage, __LINE__, __FUNCTION__, __FILE__);
    //         #endif
    //     }
    //     catch (const std::exception& e)
    //     {
    //         error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    //     }
    // }

    template<typename T>
    void ArrayCpuGpu<T>::Reshape(const int num, const int channels, const int height, const int width)
    {
        try
        {
            #ifdef USE_PYTORCH
                spImpl->pTorchBlobT->resize_({num, channels, height, width});
            #elif USE_CAFFE
                throw std::runtime_error("Not implemented Reshape");
                spImpl->pCaffeBlobT->Reshape(num, channels, height, width);
            #else
                UNUSED(num);
                UNUSED(channels);
                UNUSED(height);
                UNUSED(width);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    void ArrayCpuGpu<T>::Reshape(const std::vector<int>& shape)
    {
        try
        {
            #ifdef USE_PYTORCH
                // Convert to Long
                std::vector<long> shapeLong(shape.begin(), shape.end());
                spImpl->pTorchBlobT->resize_(shapeLong);
            #elif USE_CAFFE
                throw std::runtime_error("Not implemented Reshape");
                spImpl->pCaffeBlobT->Reshape(shape);
            #else
                UNUSED(shape);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    std::string ArrayCpuGpu<T>::shape_string() const
    {
        try
        {
            #ifdef USE_PYTORCH
                auto sizes = spImpl->pTorchBlobT->sizes();
                std::string sizeString = "[";
                for (const auto& size : sizes)
                    sizeString += std::to_string(size) + " ";
                sizeString += "]";
                return sizeString;
            #elif USE_CAFFE
                throw std::runtime_error("Not implemented shape_string");
                return spImpl->pCaffeBlobT->shape_string();
            #else
                return "";
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return "";
        }
    }

    std::vector<int> DUMB_VECTOR;
    template<typename T>
    const std::vector<int>& ArrayCpuGpu<T>::shape() const
    {
        try
        {
            #ifdef USE_PYTORCH
                torch::ArrayRef<int64_t> sizeTorch = spImpl->pTorchBlobT->sizes();
                std::vector<int> sizes(sizeTorch.begin(), sizeTorch.end());
                return sizes;
            #elif USE_CAFFE
                throw std::runtime_error("Not implemented shape1");
                return spImpl->pCaffeBlobT->shape();
            #else
                return DUMB_VECTOR;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return DUMB_VECTOR;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::shape(const int index) const
    {
        try
        {
            #ifdef USE_PYTORCH
                return spImpl->pTorchBlobT->size(index);
            #elif USE_CAFFE
                throw std::runtime_error("Not implemented shape2");
                return spImpl->pCaffeBlobT->shape(index);
            #else
                UNUSED(index);
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::num_axes() const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented num_axes");
                //return spImpl->pCaffeBlobT->num_axes();
            #else
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::count() const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented count");
                //return spImpl->pCaffeBlobT->count();
            #else
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::count(const int start_axis, const int end_axis) const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented count");
                //return spImpl->pCaffeBlobT->count(start_axis, end_axis);
            #else
                UNUSED(start_axis);
                UNUSED(end_axis);
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::count(const int start_axis) const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented count");
                //return spImpl->pCaffeBlobT->count(start_axis);
            #else
                UNUSED(start_axis);
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::CanonicalAxisIndex(const int axis_index) const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented CanonicalAxisIndex");
                //return spImpl->pCaffeBlobT->CanonicalAxisIndex(axis_index);
            #else
                UNUSED(axis_index);
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::num() const
    {
        try
        {
            #ifdef USE_PYTORCH
                return spImpl->pTorchBlobT->size(0);
            #elif USE_CAFFE
                return spImpl->pCaffeBlobT->num();
            #else
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::channels() const
    {
        try
        {
            #ifdef USE_PYTORCH
                return spImpl->pTorchBlobT->size(1);
            #elif USE_CAFFE
                return spImpl->pCaffeBlobT->channels();
            #else
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::height() const
    {
        try
        {
            #ifdef USE_PYTORCH
                return spImpl->pTorchBlobT->size(2);
            #elif USE_CAFFE
                return spImpl->pCaffeBlobT->height();
            #else
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::width() const
    {
        try
        {
            #ifdef USE_PYTORCH
                return spImpl->pTorchBlobT->size(3);
            #elif USE_CAFFE
                return spImpl->pCaffeBlobT->width();
            #else
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::LegacyShape(const int index) const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented LegacyShape");
                //return spImpl->pCaffeBlobT->LegacyShape(index);
            #else
                UNUSED(index);
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    template<typename T>
    int ArrayCpuGpu<T>::offset(const int n, const int c, const int h, const int w) const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented offset");
                //return spImpl->pCaffeBlobT->offset(n, c, h, w);
            #else
                UNUSED(n);
                UNUSED(c);
                UNUSED(h);
                UNUSED(w);
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    // template<typename T>
    // int ArrayCpuGpu<T>::offset(const std::vector<int>& indices) const
    // {
    //     try
    //     {
    //         #ifdef USE_CAFFE
    //             return spImpl->pCaffeBlobT->offset(indices);
    //         #else
    //             UNUSED(indices);
    //             return -1;
    //         #endif
    //     }
    //     catch (const std::exception& e)
    //     {
    //         error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    //         return -1;
    //     }
    // }

    template<typename T>
    T ArrayCpuGpu<T>::data_at(const int n, const int c, const int h, const int w) const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented data_at");
                //return spImpl->pCaffeBlobT->data_at(n, c, h, w);
            #else
                UNUSED(n);
                UNUSED(c);
                UNUSED(h);
                UNUSED(w);
                return T{0};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T{0};
        }
    }

    template<typename T>
    T ArrayCpuGpu<T>::diff_at(const int n, const int c, const int h, const int w) const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented diff_at");
                //return spImpl->pCaffeBlobT->diff_at(n, c, h, w);
            #else
                UNUSED(n);
                UNUSED(c);
                UNUSED(h);
                UNUSED(w);
                return T{0};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T{0};
        }
    }

    // template<typename T>
    // T ArrayCpuGpu<T>::data_at(const std::vector<int>& index) const
    // {
    //     try
    //     {
    //         #ifdef USE_CAFFE
    //             return spImpl->pCaffeBlobT->data_at(index);
    //         #else
    //             UNUSED(index);
    //             return T{0};
    //         #endif
    //     }
    //     catch (const std::exception& e)
    //     {
    //         error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    //         return T{0};
    //     }
    // }

    // template<typename T>
    // T ArrayCpuGpu<T>::diff_at(const std::vector<int>& index) const
    // {
    //     try
    //     {
    //         #ifdef USE_CAFFE
    //             return spImpl->pCaffeBlobT->diff_at(index);
    //         #else
    //             UNUSED(index);
    //             return T{0};
    //         #endif
    //     }
    //     catch (const std::exception& e)
    //     {
    //         error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    //         return T{0};
    //     }
    // }

    template<typename T>
    const T* ArrayCpuGpu<T>::cpu_data() const
    {
        try
        {
            #ifdef USE_PYTORCH
                return (T*)spImpl->pTorchBlobT->cpu().data_ptr();
            #elif USE_CAFFE
                return spImpl->pCaffeBlobT->cpu_data();
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

    template<typename T>
    void ArrayCpuGpu<T>::set_cpu_data(T* data)
    {
        try
        {
            #if USE_CAFFE
                throw std::runtime_error("Not implemented set_cpu_data");
                //spImpl->pCaffeBlobT->set_cpu_data(data);
            #else
                UNUSED(data);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    const int* ArrayCpuGpu<T>::gpu_shape() const
    {
        try
        {
            #if defined(USE_CAFFE) && (defined(USE_CUDA) || defined(USE_OPENCL))
                throw std::runtime_error("Not implemented gpu_shape");
                //return spImpl->pCaffeBlobT->gpu_shape();
            #else
                error("Required `USE_CAFFE` and `USE_CUDA` flags enabled.", __LINE__, __FUNCTION__, __FILE__);
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    template<typename T>
    const T* ArrayCpuGpu<T>::gpu_data() const
    {
        try
        {
            #ifdef USE_PYTORCH
                return (T*)spImpl->pTorchBlobT->cuda().data_ptr();
            #elif defined(USE_CAFFE) && (defined(USE_CUDA) || defined(USE_OPENCL))
                return spImpl->pCaffeBlobT->gpu_data();
            #else
                error("Required `USE_CAFFE` and `USE_CUDA` flags enabled.", __LINE__, __FUNCTION__, __FILE__);
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    template<typename T>
    void ArrayCpuGpu<T>::set_gpu_data(T* data)
    {
        try
        {
            #if defined(USE_CAFFE) && (defined(USE_CUDA) || defined(USE_OPENCL))
                throw std::runtime_error("Not implemented set_gpu_data");
                //spImpl->pCaffeBlobT->set_gpu_data(data);
            #else
                UNUSED(data);
                error("Required `USE_CAFFE` and `USE_CUDA` flags enabled.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    const T* ArrayCpuGpu<T>::cpu_diff() const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented cpu_diff");
                //return spImpl->pCaffeBlobT->cpu_diff();
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

    template<typename T>
    const T* ArrayCpuGpu<T>::gpu_diff() const
    {
        try
        {
            #if defined(USE_CAFFE) && (defined(USE_CUDA) || defined(USE_OPENCL))
                throw std::runtime_error("Not implemented gpu_diff");
                //return spImpl->pCaffeBlobT->gpu_diff();
            #else
                error("Required `USE_CAFFE` and `USE_CUDA` flags enabled.", __LINE__, __FUNCTION__, __FILE__);
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    template<typename T>
    T* ArrayCpuGpu<T>::mutable_cpu_data()
    {
        try
        {
            #ifdef USE_PYTORCH
                return (T*)spImpl->pTorchBlobT->cpu().data_ptr();
            #elif USE_CAFFE
                return spImpl->pCaffeBlobT->mutable_cpu_data();
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

    template<typename T>
    T* ArrayCpuGpu<T>::mutable_gpu_data()
    {
        try
        {
            #ifdef USE_PYTORCH
                return (T*)spImpl->pTorchBlobT->cuda().data_ptr();
            #elif defined(USE_CAFFE) && (defined(USE_CUDA) || defined(USE_OPENCL))
                return spImpl->pCaffeBlobT->mutable_gpu_data();
            #else
                error("Required `USE_CAFFE` and `USE_CUDA` flags enabled.", __LINE__, __FUNCTION__, __FILE__);
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    template<typename T>
    T* ArrayCpuGpu<T>::mutable_cpu_diff()
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented mutable_cpu_diff");
                //return spImpl->pCaffeBlobT->mutable_cpu_diff();
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

    template<typename T>
    T* ArrayCpuGpu<T>::mutable_gpu_diff()
    {
        try
        {
            #if defined(USE_CAFFE) && (defined(USE_CUDA) || defined(USE_OPENCL))
                throw std::runtime_error("Not implemented mutable_gpu_diff");
                //return spImpl->pCaffeBlobT->mutable_gpu_diff();
            #else
                error("Required `USE_CAFFE` and `USE_CUDA` flags enabled.", __LINE__, __FUNCTION__, __FILE__);
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    template<typename T>
    void ArrayCpuGpu<T>::Update()
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented Update");
                //spImpl->pCaffeBlobT->Update();
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    T ArrayCpuGpu<T>::asum_data() const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented asum_data");
                //return spImpl->pCaffeBlobT->asum_data();
            #else
                return T{0};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T{0};
        }
    }

    template<typename T>
    T ArrayCpuGpu<T>::asum_diff() const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented asum_diff");
                //return spImpl->pCaffeBlobT->asum_diff();
            #else
                return T{0};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T{0};
        }
    }

    template<typename T>
    T ArrayCpuGpu<T>::sumsq_data() const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented sumsq_data");
                //return spImpl->pCaffeBlobT->asum_data();
            #else
                return T{0};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T{0};
        }
    }

    template<typename T>
    T ArrayCpuGpu<T>::sumsq_diff() const
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented sumsq_diff");
                //return spImpl->pCaffeBlobT->asum_data();
            #else
                return T{0};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T{0};
        }
    }

    template<typename T>
    void ArrayCpuGpu<T>::scale_data(const T scale_factor)
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented scale_data");
                //spImpl->pCaffeBlobT->scale_data(scale_factor);
            #else
                UNUSED(scale_factor);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    void ArrayCpuGpu<T>::scale_diff(const T scale_factor)
    {
        try
        {
            #ifdef USE_CAFFE
                throw std::runtime_error("Not implemented scale_diff");
                //spImpl->pCaffeBlobT->scale_diff(scale_factor);
            #else
                UNUSED(scale_factor);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_FLOATING_INT_TYPES_CLASS(ArrayCpuGpu);
}
