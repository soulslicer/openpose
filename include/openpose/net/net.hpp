#ifndef OPENPOSE_NET_NET_HPP
#define OPENPOSE_NET_NET_HPP

#include <openpose/core/common.hpp>

namespace op
{
    class OP_API Net
    {
    public:
        virtual ~Net(){}

        virtual void initializationOnThread() = 0;

        virtual void forwardPass(const Array<float>& inputData) const = 0;

        virtual void forwardPass() const = 0;

        virtual void reshape(const std::vector<int>& dimensions, std::string name, bool reshape=0) const = 0;

        virtual const std::vector<int> shape(std::string name) const = 0;

        virtual std::shared_ptr<ArrayCpuGpu<float>> getOutputBlobArray() const = 0;

        virtual std::shared_ptr<ArrayCpuGpu<float>> getBlobArray(std::string name) const = 0;
    };
}

#endif // OPENPOSE_NET_NET_HPP
