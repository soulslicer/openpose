#ifndef OPENPOSE_POSE_POSE_TRACKER_HPP
#define OPENPOSE_POSE_POSE_TRACKER_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>
#include <openpose/core/keepTopNPeople.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/poseExtractorNet.hpp>
#include <openpose/tracking/personIdExtractor.hpp>
#include <openpose/tracking/personTracker.hpp>
#include <openpose/net/bodyPartConnectorBase.hpp>

namespace op
{
    struct Tracklet{
        op::Array<float> kp, kp_prev;
        int kp_hitcount;
        int kp_count;
        bool valid;
    };

    class OP_API PoseTracker
    {
    public:
        const float mRenderThreshold = 0.05;
        int mFrameCount = -1;
        std::vector<int> mTafPartPairs;
        int* mGpuTafPartPairsPtr;
        std::map<int, Tracklet> mTracklets;
        PoseModel mPoseModel;
        int mTafModel;
        int mTotalKeypoints;

        PoseTracker(PoseModel poseModel, int tafModel);

        void run(op::Array<float>& poseKeypoints,
                 const std::shared_ptr<ArrayCpuGpu<float>> tafsBlob,
                 float scale);

        virtual ~PoseTracker();

        Array<long long> getPoseIds()
        {
            Array<long long> mPoseIds;
            std::vector<long long> ids;
            for (auto& kv : mTracklets) {
                ids.emplace_back(kv.first);
            }
            mPoseIds.reset(ids.size());
            for(size_t i=0; i<ids.size(); i++){
                mPoseIds.at(i) = ids[i];
            }
            return mPoseIds;
        }

        Array<float> getPoseKeypoints()
        {
            op::Array<float> poseKeypoints({(int)mTracklets.size(), mTotalKeypoints, 3},0.0f);
            int i=0;
            for (auto& kv : mTracklets) {
                for(int j=0; j<poseKeypoints.getSize(1); j++)
                    for(int k=0; k<poseKeypoints.getSize(2); k++)
                        poseKeypoints.at({i,j,k}) = kv.second.kp.at({j,k});
                i+=1;
            }
            return poseKeypoints;
        }

        DELETE_COPY(PoseTracker);

    private:

        std::pair<op::Array<float>, std::map<int, int>> tafKernel(op::Array<float>& poseKeypoints, const std::shared_ptr<ArrayCpuGpu<float>> heatMapsBlob, float scale);

        std::vector<int> computeTrackScore(op::Array<float>& poseKeypoints, int pid, std::pair<op::Array<float>, std::map<int, int>>& tafScores);

        int getNextID(){
            int max = -1;
            for ( const auto &t : mTracklets ) {
                if(t.first > max) max = t.first;
            }
            return (max + 1) % 999;
        }

        int addNewTracklet(op::Array<float>& personKp){
            int id = getNextID();
            mTracklets[id] = Tracklet();
            mTracklets[id].kp = personKp.clone();
            mTracklets[id].kp_hitcount = mFrameCount;
            mTracklets[id].kp_count = 0;
            mTracklets[id].valid = true;
            return id;
        }

        void updateTracklet(int tid, op::Array<float>& personKp){
            mTracklets[tid].kp_prev = mTracklets[tid].kp.clone();
            mTracklets[tid].kp = personKp.clone();
            mTracklets[tid].kp_hitcount += 1;
            mTracklets[tid].kp_count += 1;
        }

        void reset(){
            mFrameCount = -1;
            mTracklets.clear();
        }
    };
}

#endif // OPENPOSE_POSE_POSE_TRACKER_HPP
