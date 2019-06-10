#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <openpose/utilities/fastMath.hpp>
#endif
#include <openpose/pose/poseTracker.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace op
{
    cv::Mat mat_from_blob3(std::shared_ptr<ArrayCpuGpu<float>> src, int channel)
    {
        cv::Mat m(src->shape()[2], src->shape()[3], CV_32FC1, (float*)(src->cpu_data() + channel*src->shape()[2]*src->shape()[3]));
        return m;
    }

    int getValidKps(op::Array<float>& person_kp, float render_threshold){
        int valid = 0;
        for(int i=0; i<person_kp.getSize(0); i++){
            if(person_kp.at({i,2}) > render_threshold) valid += 1;
        }

        return valid;
    }

    void rescaleKp(op::Array<float>& person_kp, float scal){
        for(int i=0; i<person_kp.getSize(0); i++){
            person_kp.at({i,0}) *= scal;
            person_kp.at({i,1}) *= scal;
        }
    }

    std::pair<int, int> mostCommon(std::vector<int>& lst, int exclude=-1){
        std::map<int, int> mydict = {};
        int cnt = 0;
        int itm = 0;  // in Python you made this a string '', which seems like a bug

        for (auto&& item : lst) {
            if(item == -1) continue;
            mydict[item] = mydict.emplace(item, 0).first->second + 1;
            if (mydict[item] >= cnt) {
                std::tie(cnt, itm) = std::tie(mydict[item], item);
            }
        }

        return std::pair<int, int>(itm, cnt);
    }

    op::Array<float> getPerson(op::Array<float>& poseKeypoints, int pid){
        return op::Array<float>({poseKeypoints.getSize()[1], poseKeypoints.getSize()[2]}, poseKeypoints.getPtr() + pid*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2]);
    }

    bool pairSort(const std::pair<int, int>& struct1, const std::pair<int, int>& struct2)
    {
        return (struct1.first < struct2.first);
    }

    PoseTracker::PoseTracker(PoseModel poseModel, int tafModel)
    {
        mPoseModel = poseModel;
        mTafModel = tafModel;

        mTotalKeypoints = getPoseBodyPartMapping(poseModel).size()-1;
        if(poseModel == PoseModel::BODY_25B) mTotalKeypoints+=1;

        mTafPartPairs = getTafPartMapping(mTafModel);

        cudaMalloc((void **)&mGpuTafPartPairsPtr, mTafPartPairs.size() * sizeof(int));
        cudaMemcpy(mGpuTafPartPairsPtr, &mTafPartPairs[0], mTafPartPairs.size() * sizeof(int),
                   cudaMemcpyHostToDevice);

        mTrackVelocity = true;
    }

    PoseTracker::~PoseTracker()
    {
    }

    std::vector<int> PoseTracker::computeTrackScore(op::Array<float>& poseKeypoints, int pid, std::pair<op::Array<float>, std::map<int, int>>& tafScores)
    {
        op::Array<float> personKp = getPerson(poseKeypoints, pid);
        std::vector<int> finalIdxs(personKp.getSize()[0], -1);
        std::vector<float> finalScores(personKp.getSize()[0], -1);

        //for(int i=25; i<mTafPartPairs.size()/2; i++){
        //for(int i=0; i<25; i++){
        for(int i=0; i<mTafPartPairs.size()/2; i++){
            auto partA = mTafPartPairs[i*2];
            auto partB = mTafPartPairs[i*2 + 1];

            // Ignore Foot?
            if(partA == 19 || partB == 19 ||
               partA == 20 || partB == 20 ||
                partA == 21 || partB == 21 ||
                partA == 22 || partB == 22 ||
                partA == 23 || partB == 23 ||
                partA == 24 || partB == 24) continue;

            if(personKp.at({partA, 2}) < mRenderThreshold) continue;

            int best_tid = -1;
            float best_fscore = 0;
            for ( auto &kv : tafScores.second ){
                int tid = kv.first;
                int tid_map = kv.second;
                Tracklet& tracklet = mTracklets[tid];
                if(tracklet.valid == false) throw std::runtime_error("Should not go here");
                if(tracklet.kp.at({partB, 2}) < mRenderThreshold) continue;
                auto fscore = tafScores.first.at({i, pid, tid_map});

                if(fscore > best_fscore){
                    best_fscore = fscore;
                    best_tid = tid;
                }
            }

            if(best_tid >= 0){
                if(finalIdxs[partA] != -1){
                    if(best_fscore > finalScores[partA]){
                        finalIdxs[partA]=best_tid;
                        finalScores[partA]=best_fscore;
                    }
                }else{
                    finalIdxs[partA]=best_tid;
                    finalScores[partA]=best_fscore;
                }

                //finalIdxs[partA]=best_tid;
                //finalScores[partA]=best_fscore;

            }

        }

        return finalIdxs;
    }

    std::pair<op::Array<float>, std::map<int, int>> PoseTracker::tafKernel(op::Array<float>& poseKeypoints, const std::shared_ptr<ArrayCpuGpu<float>> tafsBlob, float scale)
    {
        std::map<int, int> tidToMap;
        op::Array<float> trackletKeypoints({getValidTrackletsCount(), poseKeypoints.getSize(1), poseKeypoints.getSize(2)},0.0f);
        int i=0;
        for (auto& kv : mTracklets) {
            if(!kv.second.valid) continue;
            for(int j=0; j<poseKeypoints.getSize(1); j++)
                for(int k=0; k<poseKeypoints.getSize(2); k++)
                    trackletKeypoints.at({i,j,k}) = kv.second.kp.at({j,k});
            tidToMap[kv.first] = i;
            i+=1;
        }

        op::Array<float> tafScores; // pairs/ pose/ tracklet
        op::tafScoreGPU(poseKeypoints, trackletKeypoints, tafsBlob, tafScores, mTafPartPairs, mGpuTafPartPairsPtr, 0, scale);

        return std::pair<op::Array<float>, std::map<int, int>>(tafScores, tidToMap);
    }

    void PoseTracker::run(op::Array<float>& poseKeypoints,
             const std::shared_ptr<ArrayCpuGpu<float>> tafsBlob,
             float scale)
    {
        bool debug=false;
        //if(mFrameCount > 1000) debug=true;
        if(debug) std::this_thread::sleep_for (std::chrono::seconds(1));

        if(!poseKeypoints.getSize(0)) return;
        mFrameCount += 1;

        // Update Params
        auto to_update_set = std::map<int, std::vector<std::pair<int, int>>>();
        auto tid_updated = std::vector<int>();
        auto tid_added = std::vector<int>();

        // Kernel goes here
        std::pair<op::Array<float>, std::map<int, int>> tafScores = tafKernel(poseKeypoints, tafsBlob, scale);

        // Iterate Pose Keypoints (Global Score)
        for(int i=0; i<poseKeypoints.getSize()[0]; i++){
            op::Array<float> personKp = getPerson(poseKeypoints, i);
            // Score
            auto finalIdxs = computeTrackScore(poseKeypoints, i, tafScores);

            if(debug){
            for(auto item : finalIdxs) std::cout << item << " ";
            std::cout << std::endl;
            }

            auto mc = mostCommon(finalIdxs);
            auto mostCommonIdx = mc.first; auto mostCommonCount = mc.second;

            if(mostCommonCount >= 5){
                if(!to_update_set.count(mostCommonIdx)) to_update_set[mostCommonIdx] = {};
                to_update_set[mostCommonIdx].emplace_back(std::pair<int, int>(mostCommonCount,i));
                //if(debug) std::cout << "Set: " << mostCommonIdx << " c: " << mostCommonCount << " i: " << i << std::endl;
            }else{
                if(getValidKps(personKp, mRenderThreshold) <= 5) continue;
                //if(frame_count < 2){
                int newId = addNewTracklet(personKp);
                tid_added.emplace_back(newId);
                //std::cout << "Add : " << newId << std::endl;
                //}
            }
        }

        // Global Update
        for (auto& kv : to_update_set) {
            auto mostCommonIdx = kv.first;
            auto& item = kv.second;
            if(item.size() > 1){
                std::sort(item.begin(), item.end(), pairSort);
                auto best_item_index = item.back().second;
                auto best_person_kp = getPerson(poseKeypoints, best_item_index);
                updateTracklet(mostCommonIdx, best_person_kp);
                if(debug) std::cout << "Update : " << best_item_index << " into tracklet " << mostCommonIdx << std::endl;
                tid_updated.emplace_back(mostCommonIdx);

                item.pop_back();
                for(auto& remain_item : item){
                    auto personKp = getPerson(poseKeypoints, remain_item.second);
                    if(getValidKps(personKp, mRenderThreshold) <= 5) continue;
                    int newId = addNewTracklet(personKp);
                    //std::cout << "Add : " << newId << std::endl;
                    tid_added.emplace_back(newId);
                }
            }else{
                auto best_person_kp = getPerson(poseKeypoints, item[0].second);
                updateTracklet(mostCommonIdx, best_person_kp);
                //if(debug) std::cout << "Update : " << item[0].second << " into tracklet " << mostCommonIdx << std::endl;
                tid_updated.emplace_back(mostCommonIdx);
            }
        }

        // Deletion
        std::vector<int> to_delete;
        for (auto& kv : mTracklets) {
            auto tidx = kv.first;
            auto& tracklet = kv.second;
            if(tracklet.kp_hitcount - mFrameCount < 0) {
                tracklet.valid = false;
                to_delete.emplace_back(tidx);
                //if(debug) std::cout << "Delete : " << tidx << std::endl;
            }
            //if(tracklet.kp_hitcount - mFrameCount < -5) {
            //    to_delete.emplace_back(tidx);
            //}
            if(tracklet.kp.getSize(1) == 0) throw std::runtime_error("Track Error");
        }
        for(auto to_del : to_delete) mTracklets.erase(mTracklets.find(to_del));
    }

}

// VIZ
//        std::cout << poseKeypoints << std::endl;
//        std::cout << getPoseKeypoints() << std::endl;

//        if(debug){
//            int mindex = 36;
//            cv::Mat hm = cv::abs(mat_from_blob3(tafsBlob, mindex*2)) + cv::abs(mat_from_blob3(tafsBlob, mindex*2 + 1));
//            cv::cvtColor(hm, hm, cv::COLOR_GRAY2BGR);

//            cv::Mat fx = mat_from_blob3(tafsBlob, mindex*2);
//            cv::Mat fy = mat_from_blob3(tafsBlob, mindex*2 + 1);
//            for(int u=0; u<fx.size().width; u+=10){
//                for(int v=0; v<fx.size().height; v+=10){
//                    cv::Point2f p1(u,v);
//                    cv::Point2i currP(u,v);
//                    cv::Point2f p2(u + 20*fx.at<float>(currP), v + 20*fy.at<float>(currP));
//                    cv::line(hm, p1, p2, cv::Scalar(0,1,0));
//                }
//            }

//            cv::imshow("win", hm);
//            cv::waitKey(15);
//        }

/////////////////
