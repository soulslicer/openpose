#include <openpose/gpu/cuda.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/pose/bodyPartConnectorBase.hpp>
#include <iostream>

namespace op
{
    using clock_value_t = long long;

    __device__ void sleep(clock_value_t sleep_cycles)
    {
        clock_value_t start = clock64();
        clock_value_t cycles_elapsed;
        do { cycles_elapsed = clock64() - start; }
        while (cycles_elapsed < sleep_cycles);
    }

    template <typename T>
    __global__ void bpcKernel(const T* heatMapPtr, const T* peaksPtrA, const T* peaksPtrB, const unsigned int* bodyPartPairsPtr, const unsigned int* mapIdxPtr, const int POSE_MAX_PEOPLE, const int TOTAL_BODY_PARTS, const int heatmapWidth, const int heatmapHeight)
    {
        const auto i = (blockIdx.x * blockDim.x) + threadIdx.x;
        const auto j = (blockIdx.y * blockDim.y) + threadIdx.y;
        const auto k = (blockIdx.z * blockDim.z) + threadIdx.z;

        int partA = bodyPartPairsPtr[i*2];
        int partB = bodyPartPairsPtr[i*2 + 1];
        int mapIdxX = mapIdxPtr[i*2];
        int mapIdxY = mapIdxPtr[i*2 + 1];

        const T* bodyPartA = peaksPtrA + (partA*POSE_MAX_PEOPLE*3 + j*3);
        const T* bodyPartB = peaksPtrB + (partB*POSE_MAX_PEOPLE*3 + k*3);
        const T* mapX = heatMapPtr + mapIdxX*heatmapWidth*heatmapHeight;
        const T* mapY = heatMapPtr + mapIdxY*heatmapWidth*heatmapHeight;

        if(bodyPartA[2] < 0.05 || bodyPartB[2] < 0.05) return;



        //sleep(1000);

        if(j==0 && k==0){
            //printf("%d \n", i);
            //printf("%d %d %d \n",bodyPartPairsPtr[0],bodyPartPairsPtr[1],bodyPartPairsPtr[2]);
        }
        //printf("%d %d %d \n",x,y,z);
    }

    template <typename T>
    void connectBodyPartsGpu(Array<T>& poseKeypoints, Array<T>& poseScores, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                             const int minSubsetCnt, const T minSubsetScore, const T scaleFactor,
                             const T* const heatMapGpuPtr, const T* const peaksGpuPtr)
    {
        try
        {
            // Parts Connection
            const auto& bodyPartPairs = getPosePartPairs(poseModel);
            const auto& mapIdx = getPoseMapIndex(poseModel);
            const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
            const auto numberBodyPartPairs = bodyPartPairs.size() / 2;

            // Vector<int> = Each body part + body parts counter; double = subsetScore
            std::vector<std::pair<std::vector<int>, double>> subset;
            const auto subsetCounterIndex = numberBodyParts;
            const auto subsetSize = numberBodyParts+1;

            const auto peaksOffset = 3*(maxPeaks+1);
            const auto heatMapOffset = heatMapSize.area();

            const dim3 threadsPerBlock{numberBodyPartPairs, 1, 1};
            const dim3 numBlocks{1, POSE_MAX_PEOPLE, POSE_MAX_PEOPLE};

            unsigned int* bodyPartPairsGpuPtr;
            cudaMallocHost((void **)&bodyPartPairsGpuPtr, bodyPartPairs.size() * sizeof(unsigned int));
            cudaMemcpy(bodyPartPairsGpuPtr, &bodyPartPairs[0], bodyPartPairs.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

            unsigned int* mapIdxGpuPtr;
            cudaMallocHost((void **)&mapIdxGpuPtr, mapIdx.size() * sizeof(unsigned int));
            cudaMemcpy(mapIdxGpuPtr, &mapIdx[0], mapIdx.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

            bpcKernel<<<numBlocks, threadsPerBlock>>>(heatMapGpuPtr, peaksGpuPtr, peaksGpuPtr, bodyPartPairsGpuPtr, mapIdxGpuPtr, POSE_MAX_PEOPLE, numberBodyParts, heatMapSize.x, heatMapSize.y);

            cudaThreadSynchronize();

            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void connectBodyPartsGpu(Array<float>& poseKeypoints, Array<float>& poseScores,
                                      const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
                                      const float interMinAboveThreshold, const float interThreshold,
                                      const int minSubsetCnt, const float minSubsetScore, const float scaleFactor,
                                      const float* const heatMapGpuPtr, const float* const peaksGpuPtr);
    template void connectBodyPartsGpu(Array<double>& poseKeypoints, Array<double>& poseScores,
                                      const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
                                      const double interMinAboveThreshold, const double interThreshold,
                                      const int minSubsetCnt, const double minSubsetScore, const double scaleFactor,
                                      const double* const heatMapGpuPtr, const double* const peaksGpuPtr);
}
