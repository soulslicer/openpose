#include <limits> // std::numeric_limits
#ifdef USE_CAFFE
    #include <caffe/blob.hpp>
#endif
#include <openpose/gpu/cuda.hpp>
#include <openpose/net/bodyPartConnectorCaffe.hpp>
#include <openpose/net/maximumCaffe.hpp>
#include <openpose/net/netCaffe.hpp>
#include <openpose/net/netOpenCv.hpp>
#include <openpose/net/nmsCaffe.hpp>
#include <openpose/net/resizeAndMergeCaffe.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/pose/poseExtractorCaffe.hpp>
#include <caffe/caffe.hpp>
#include <openpose/net/bodyPartConnectorBase.hpp>
#include <openpose/core/scaleAndSizeExtractor.hpp>

#define MODEL_PATH "/home/raaj/openpose/tracker/"
#define FLAGS_render_threshold 0.05

namespace op
{
    const bool TOP_DOWN_REFINEMENT = false; // Note: +5% acc 1 scale, -2% max acc setting

    void gpu_copy(boost::shared_ptr<caffe::Blob<float>> dest, boost::shared_ptr<caffe::Blob<float>> src){
        size_t size = sizeof(float)*src->shape()[1]*src->shape()[2]*src->shape()[3];
        cudaMemcpy(dest->mutable_gpu_data(),src->mutable_gpu_data(),size, cudaMemcpyDeviceToDevice);
    }
    void gpu_copy(std::shared_ptr<caffe::Blob<float>> dest, boost::shared_ptr<caffe::Blob<float>> src){
        size_t size = sizeof(float)*src->shape()[1]*src->shape()[2]*src->shape()[3];
        cudaMemcpy(dest->mutable_gpu_data(),src->mutable_gpu_data(),size, cudaMemcpyDeviceToDevice);
    }
    void gpu_copy(boost::shared_ptr<caffe::Blob<float>> dest, std::shared_ptr<caffe::Blob<float>> src){
        size_t size = sizeof(float)*src->shape()[1]*src->shape()[2]*src->shape()[3];
        cudaMemcpy(dest->mutable_gpu_data(),src->mutable_gpu_data(),size, cudaMemcpyDeviceToDevice);
    }

    int getValidKps(op::Array<float>& person_kp, float render_threshold){
        int valid = 0;
        for(int i=0; i<person_kp.getSize(0); i++){
            if(person_kp.at({i,2}) > render_threshold) valid += 1;
        }

        return valid;
    }

    void rescale_kp(op::Array<float>& person_kp, float scal){
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

    op::Array<float> get_person_no_copy(op::Array<float>& poseKeypoints, int pid){
        return op::Array<float>({poseKeypoints.getSize()[1], poseKeypoints.getSize()[2]}, poseKeypoints.getPtr() + pid*poseKeypoints.getSize()[1]*poseKeypoints.getSize()[2]);
    }

    op::Array<float> get_bp_no_copy(op::Array<float>& person_kp, int bp){
        return op::Array<float>({person_kp.getSize()[1]}, person_kp.getPtr() + bp*person_kp.getSize()[1]);
    }

    float l2(op::Array<float>&a, op::Array<float>&b){
        return sqrt(pow(b[0]-a[0],2) + pow(b[1]-a[1],2));
    }

    struct Tracklet{
        op::Array<float> kp, kp_prev;
        int kp_hitcount;
        int kp_count;
        bool valid;
    };

    bool pair_sort(const std::pair<int, int>& struct1, const std::pair<int, int>& struct2)
    {
        return (struct1.first < struct2.first);
    }

    class Tracker{
    public:
        const float render_threshold = FLAGS_render_threshold;
        std::vector<int> taf_part_pairs = {1,8, 9,10, 10,11, 8,9, 8,12, 12,13, 13,14, 1,2, 2,3, 3,4, 2,17, 1,5, 5,6, 6,7, 5,18, 1,0, 0,15, 0,16, 15,17, 16,18, 1,19, 19,20, 5,12, 2,9};
        // for heatmaps offset = 22+48 --- i*(48) + [j*2 + k]

        int* gpu_taf_part_pairs_ptr = nullptr;

        std::map<int, Tracklet> tracklets_internal;
        int frame_count = -1;

        int get_next_id(){
            int max = -1;
            for ( const auto &t : tracklets_internal ) {
                if(t.first > max) max = t.first;
            }
            return (max + 1) % 999;
        }

        int add_new_tracklet(op::Array<float>& person_kp){
            int id = get_next_id();
            tracklets_internal[id] = Tracklet();
            tracklets_internal[id].kp = person_kp.clone();
            tracklets_internal[id].kp_hitcount = frame_count;
            tracklets_internal[id].kp_count = 0;
            tracklets_internal[id].valid = true;
            return id;
        }

        void update_tracklet(int tid, op::Array<float>& person_kp){
            tracklets_internal[tid].kp_prev = tracklets_internal[tid].kp.clone();
            tracklets_internal[tid].kp = person_kp.clone();
            tracklets_internal[tid].kp_hitcount += 1;
            tracklets_internal[tid].kp_count += 1;
        }

        void reset(){
            frame_count = -1;
            tracklets_internal.clear();
        }

        std::vector<int> compute_track_score(op::Array<float>& pose_keypoints, int pid, std::pair<op::Array<float>, std::map<int, int>>& taf_scores){
            op::Array<float> person_kp = get_person_no_copy(pose_keypoints, pid);
            std::vector<int> final_idxs(person_kp.getSize()[0], -1);

            for(int i=24; i<taf_part_pairs.size()/2; i++){
                auto partA = taf_part_pairs[i*2];
                auto partB = taf_part_pairs[i*2 + 1];

                if(partA == 15 || partB == 15 ||
                        partA == 16 || partB == 16 ||
                        partA == 17 || partB == 17 ||
                        partA == 18 || partB == 18) continue;

                if(person_kp.at({partA, 2}) < render_threshold) continue;

                //if(final_idxs[partA] >= 0) continue;

                int best_tid = -1;
                int best_fscore = 0;
                for ( auto &kv : taf_scores.second ){
                    int tid = kv.first;
                    int tid_map = kv.second;
                    Tracklet& tracklet = tracklets_internal[tid];
                    if(tracklet.kp.at({partB, 2}) < render_threshold) continue;
                    auto fscore = taf_scores.first.at({i, pid, tid_map});
                    if(fscore > best_fscore){
                        best_fscore = fscore;
                        best_tid = tid;
                    }
                }

                if(best_tid >= 0) final_idxs[partA]=best_tid;

            }

            return final_idxs;
        }



        std::pair<op::Array<float>, std::map<int, int>> taf_kernel(op::Array<float>& pose_keypoints, std::shared_ptr<caffe::Blob<float>> heatMapsBlob){
            // 1
            std::map<int, int> tid_to_map;
            op::Array<float> tracklet_keypoints({(int)tracklets_internal.size(), pose_keypoints.getSize(1), pose_keypoints.getSize(2)},0.0f);
            int i=0;
            for (auto& kv : tracklets_internal) {
                for(int j=0; j<pose_keypoints.getSize(1); j++)
                    for(int k=0; k<pose_keypoints.getSize(2); k++)
                        tracklet_keypoints.at({i,j,k}) = kv.second.kp.at({j,k});
                tid_to_map[kv.first] = i;
                i+=1;
            }

            op::Array<float> taf_scores;
            op::tafScoreGPU(pose_keypoints, tracklet_keypoints, heatMapsBlob, taf_scores, taf_part_pairs, gpu_taf_part_pairs_ptr, 70);

            return std::pair<op::Array<float>, std::map<int, int>>(taf_scores, tid_to_map);
        }

        void run(op::Array<float>& pose_keypoints, std::shared_ptr<caffe::Blob<float>> heatMapsBlob, std::shared_ptr<caffe::Blob<float>> peaksBlob, float scale){
            if(!pose_keypoints.getSize(0)) return;

            frame_count += 1;

            // Scale Down
            for(int i=0; i<pose_keypoints.getSize()[0]; i++){
                op::Array<float> person_kp = get_person_no_copy(pose_keypoints, i);
                rescale_kp(person_kp, scale);
            }

            // Update Params
            auto to_update_set = std::map<int, std::vector<std::pair<int, int>>>();
            auto tid_updated = std::vector<int>();
            auto tid_added = std::vector<int>();

            // Kernel goes here
            std::pair<op::Array<float>, std::map<int, int>> taf_scores = taf_kernel(pose_keypoints, heatMapsBlob);

            // Iterate Pose Keypoints (Global Score)
            for(int i=0; i<pose_keypoints.getSize()[0]; i++){
                op::Array<float> person_kp = get_person_no_copy(pose_keypoints, i);
                // Score
                auto final_idxs = compute_track_score(pose_keypoints, i, taf_scores);
                auto mc = mostCommon(final_idxs);
                auto mostCommonIdx = mc.first; auto mostCommonCount = mc.second;

                if(mostCommonCount >= 5){
                    if(!to_update_set.count(mostCommonIdx)) to_update_set[mostCommonIdx] = {};
                    to_update_set[mostCommonIdx].emplace_back(std::pair<int, int>(mostCommonCount,i));
                }else{
                    if(getValidKps(person_kp, render_threshold) <= 5) continue;
                    //if(frame_count < 2){
                    int new_id = add_new_tracklet(person_kp);
                    tid_added.emplace_back(new_id);
                    //}
                }

                //break;
            }

            // Global Update
            for (auto& kv : to_update_set) {
                auto mostCommonIdx = kv.first;
                auto& item = kv.second;
                if(item.size() > 1){
                    std::sort(item.begin(), item.end(), pair_sort);
                    auto best_item_index = item.back().second;
                    auto best_person_kp = get_person_no_copy(pose_keypoints, best_item_index);
                    update_tracklet(mostCommonIdx, best_person_kp);
                    //std::cout << "Update : " << best_item_index << " into tracklet " << mostCommonIdx << std::endl;
                    tid_updated.emplace_back(mostCommonIdx);

                    item.pop_back();
                    for(auto& remain_item : item){
                        auto person_kp = get_person_no_copy(pose_keypoints, remain_item.second);
                        if(getValidKps(person_kp, render_threshold) <= 5) continue;
                        int new_id = add_new_tracklet(person_kp);
                        //std::cout << "Add : " << new_id << std::endl;
                        tid_added.emplace_back(new_id);
                    }
                }else{
                    auto best_person_kp = get_person_no_copy(pose_keypoints, item[0].second);
                    update_tracklet(mostCommonIdx, best_person_kp);
                    //std::cout << "Update : " << item[0].second << " into tracklet " << mostCommonIdx << std::endl;
                    tid_updated.emplace_back(mostCommonIdx);
                }
            }

            // Deletion
            std::vector<int> to_delete;
            for (auto& kv : tracklets_internal) {
                auto tidx = kv.first;
                auto& tracklet = kv.second;
                if(tracklet.kp_hitcount - frame_count < 0) {
                    to_delete.emplace_back(tidx);
                    //std::cout << "Delete : " << tidx << std::endl;
                }

                if(tracklet.kp.getSize(1) == 0) std::cout << "SHIT" << std::endl;

            }
            for(auto to_del : to_delete) tracklets_internal.erase(tracklets_internal.find(to_del));

            // Sanity Check

        }

        Tracker(){

            // Make TAF Part pairs in my cross linked formulation so double
            int tpp_size = taf_part_pairs.size()/2;
            for(int i=0; i<tpp_size; i++){
                taf_part_pairs.emplace_back(taf_part_pairs[i*2 + 1]);
                taf_part_pairs.emplace_back(taf_part_pairs[i*2 + 0]);
            }
        }

    };


    struct NetSet{
        std::shared_ptr<caffe::Net<float> > netVGG, netA, netB;
        std::shared_ptr<caffe::Blob<float> > pafMem, hmMem, fmMem;
        void load(){
            // Load nets
            netVGG.reset(new caffe::Net<float>(std::string(MODEL_PATH)+"vgg.prototxt", caffe::TEST));
            netA.reset(new caffe::Net<float>(std::string(MODEL_PATH)+"pose_deploy_1.prototxt", caffe::TEST));
            netB.reset(new caffe::Net<float>(std::string(MODEL_PATH)+"pose_deploy_2.prototxt", caffe::TEST));
            netVGG->CopyTrainedLayersFrom(std::string(MODEL_PATH)+"pose_iter_264000.caffemodel");
            netA->CopyTrainedLayersFrom(std::string(MODEL_PATH)+"pose_iter_264000.caffemodel");
            netB->CopyTrainedLayersFrom(std::string(MODEL_PATH)+"pose_iter_264000.caffemodel");
            pafMem.reset(new caffe::Blob<float>());
            hmMem.reset(new caffe::Blob<float>());
            fmMem.reset(new caffe::Blob<float>());
        }

        void reshape(){

        }
    };

    struct PoseExtractorCaffe::ImplPoseExtractorCaffe
    {
        #ifdef USE_CAFFE
            // Used when increasing spNets
            const PoseModel mPoseModel;
            const int mGpuId;
            const std::string mModelFolder;
            const bool mEnableGoogleLogging;
            // General parameters
            std::vector<std::shared_ptr<Net>> spNets;
            std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
            std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
            std::shared_ptr<BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe;
            std::shared_ptr<MaximumCaffe<float>> spMaximumCaffe;
            std::vector<std::vector<int>> mNetInput4DSizes;
            std::vector<double> mScaleInputToNetInputs;
            std::shared_ptr<op::ScaleAndSizeExtractor> scaleAndSizeExtractor;

            // Init with thread
            std::vector<boost::shared_ptr<caffe::Blob<float>>> spCaffeNetOutputBlobs;
            std::shared_ptr<caffe::Blob<float>> spHeatMapsBlob;
            std::shared_ptr<caffe::Blob<float>> spPeaksBlob;
            std::shared_ptr<caffe::Blob<float>> spMaximumPeaksBlob;

            std::vector<NetSet> nets;
            std::vector<float> scales;
            bool first_frame = true;
            double scaleInputToOutput;
            float pointScale;
            op::Point<int> outputResolution;
            std::map<std::string, int> COCO21_MAPPING;
            std::vector<std::vector<int>> colors;
            Tracker tracker;

            ImplPoseExtractorCaffe(const PoseModel poseModel, const int gpuId,
                                   const std::string& modelFolder, const bool enableGoogleLogging) :
                mPoseModel{poseModel},
                mGpuId{gpuId},
                mModelFolder{modelFolder},
                mEnableGoogleLogging{enableGoogleLogging},
                spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
                spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
                spBodyPartConnectorCaffe{std::make_shared<BodyPartConnectorCaffe<float>>()},
                spMaximumCaffe{(TOP_DOWN_REFINEMENT ? std::make_shared<MaximumCaffe<float>>() : nullptr)}
            {
            }
        #endif
    };

    #ifdef USE_CAFFE
        std::vector<caffe::Blob<float>*> caffeNetSharedToPtr(
            std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob)
        {
            try
            {
                // Prepare spCaffeNetOutputBlobss
                std::vector<caffe::Blob<float>*> caffeNetOutputBlobs(caffeNetOutputBlob.size());
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

        inline void reshapePoseExtractorCaffe(std::shared_ptr<ResizeAndMergeCaffe<float>>& resizeAndMergeCaffe,
                                              std::shared_ptr<NmsCaffe<float>>& nmsCaffe,
                                              std::shared_ptr<BodyPartConnectorCaffe<float>>& bodyPartConnectorCaffe,
                                              std::shared_ptr<MaximumCaffe<float>>& maximumCaffe,
                                              std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob,
                                              std::shared_ptr<caffe::Blob<float>>& heatMapsBlob,
                                              std::shared_ptr<caffe::Blob<float>>& peaksBlob,
                                              std::shared_ptr<caffe::Blob<float>>& maximumPeaksBlob,
                                              const float scaleInputToNetInput,
                                              const PoseModel poseModel,
                                              const int gpuID)
        {
            try
            {
                // HeatMaps extractor blob and layer
                // Caffe modifies bottom - Heatmap gets resized
                const auto caffeNetOutputBlobs = caffeNetSharedToPtr(caffeNetOutputBlob);
                resizeAndMergeCaffe->Reshape(caffeNetOutputBlobs, {heatMapsBlob.get()},
                                             getPoseNetDecreaseFactor(poseModel), 1.f/scaleInputToNetInput, true,
                                             gpuID);
                // Pose extractor blob and layer
                nmsCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()}, getPoseMaxPeaks(),
                                  getPoseNumberBodyParts(poseModel), gpuID);
                // Pose extractor blob and layer
                bodyPartConnectorCaffe->Reshape({heatMapsBlob.get(), peaksBlob.get()});
                if (TOP_DOWN_REFINEMENT)
                    maximumCaffe->Reshape({heatMapsBlob.get()}, {maximumPeaksBlob.get()});
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

        void addCaffeNetOnThread(std::vector<std::shared_ptr<Net>>& net,
                                 std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob,
                                 const PoseModel poseModel, const int gpuId,
                                 const std::string& modelFolder, const bool enableGoogleLogging)
        {
            try
            {
//                // Add Caffe Net
//                net.emplace_back(
//                    std::make_shared<NetCaffe>(
//                        modelFolder + getPoseProtoTxt(poseModel),
//                        modelFolder + getPoseTrainedModel(poseModel),
//                        gpuId, enableGoogleLogging));
//                // net.emplace_back(
//                //     std::make_shared<NetOpenCv>(
//                //         modelFolder + getPoseProtoTxt(poseModel),
//                //         modelFolder + getPoseTrainedModel(poseModel),
//                //         gpuId));
//                // UNUSED(enableGoogleLogging);
//                // Initializing them on the thread
//                net.back()->initializationOnThread();
//                caffeNetOutputBlob.emplace_back(((NetCaffe*)net.back().get())->getOutputBlob());
//                // caffeNetOutputBlob.emplace_back(((NetOpenCv*)net.back().get())->getOutputBlob());
//                // Sanity check
//                if (net.size() != caffeNetOutputBlob.size())
//                    error("Weird error, this should not happen. Notify us.", __LINE__, __FUNCTION__, __FILE__);
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

    PoseExtractorCaffe::PoseExtractorCaffe(const PoseModel poseModel, const std::string& modelFolder,
                                           const int gpuId, const std::vector<HeatMapType>& heatMapTypes,
                                           const ScaleMode heatMapScale, const bool addPartCandidates,
                                           const bool maximizePositives, const bool enableGoogleLogging) :
        PoseExtractorNet{poseModel, heatMapTypes, heatMapScale, addPartCandidates, maximizePositives}
        #ifdef USE_CAFFE
        , upImpl{new ImplPoseExtractorCaffe{poseModel, gpuId, modelFolder, enableGoogleLogging}}
        #endif
    {
        try
        {
            #ifdef USE_CAFFE
                // Layers parameters
                upImpl->spBodyPartConnectorCaffe->setPoseModel(upImpl->mPoseModel);
                upImpl->spBodyPartConnectorCaffe->setMaximizePositives(maximizePositives);
            #else
                UNUSED(poseModel);
                UNUSED(modelFolder);
                UNUSED(gpuId);
                UNUSED(heatMapTypes);
                UNUSED(heatMapScale);
                UNUSED(addPartCandidates);
                error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PoseExtractorCaffe::~PoseExtractorCaffe()
    {
    }

    void PoseExtractorCaffe::netInitializationOnThread()
    {
        try
        {
            #ifdef USE_CAFFE
                // Logging
                log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Initialize Caffe net
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
                // Initialize blobs
                upImpl->spHeatMapsBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                upImpl->spPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
                if (TOP_DOWN_REFINEMENT)
                    upImpl->spMaximumPeaksBlob = {std::make_shared<caffe::Blob<float>>(1,1,1,1)};
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

    void PoseExtractorCaffe::forwardPass(const std::vector<Array<float>>& inputNetData,
                                         const Point<int>& inputDataSize,
                                         const std::vector<double>& scaleInputToNetInputs)
    {
        try
        {
            #ifdef USE_CAFFE
                // Sanity checks
                if (inputNetData.empty())
                    error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
                for (const auto& inputNetDataI : inputNetData)
                    if (inputNetDataI.empty())
                        error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
                if (inputNetData.size() != scaleInputToNetInputs.size())
                    error("Size(inputNetData) must be same than size(scaleInputToNetInputs).",
                          __LINE__, __FUNCTION__, __FILE__);

                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


                // Resize std::vectors if required
                const auto numberScales = inputNetData.size();
                upImpl->mNetInput4DSizes.resize(numberScales);
                if(upImpl->first_frame){
                    log("First Frame Load");
                    caffe::Caffe::set_mode(caffe::Caffe::GPU);
                    caffe::Caffe::SetDevice(0);
                    google::InitGoogleLogging("XXX");
                    google::SetCommandLineOption("GLOG_minloglevel", "2");
                    for(int i=0; i<numberScales; i++){
                        upImpl->nets.emplace_back(NetSet());
                        upImpl->nets.back().load();
                        upImpl->mNetInput4DSizes.at(i) = inputNetData[i].getSize();
                    }
                }

                // Run Net
                std::vector<boost::shared_ptr<caffe::Blob<float>>> caffeNetOutputBlob;
                for(int i=0; i<numberScales; i++){
                    NetSet& netSet = upImpl->nets[i];
                    const op::Array<float>& imageForNet = inputNetData[i];

                    if(upImpl->first_frame){
                        log("First Frame Reshape");

                        // Do Reshaping once
                        netSet.netVGG->blob_by_name("image")->Reshape({1, 3, imageForNet.getSize(2), imageForNet.getSize(3)});
                        netSet.netVGG->Reshape();
                        netSet.netA->blob_by_name("conv4_4_CPM")->Reshape(netSet.netVGG->blob_by_name("conv4_4_CPM")->shape());
                        netSet.netA->Reshape();
                        netSet.netB->blob_by_name("conv4_4_CPM")->Reshape(netSet.netVGG->blob_by_name("conv4_4_CPM")->shape());
                        netSet.netB->blob_by_name("last_paf")->Reshape(netSet.netA->blob_by_name("Mconv10_stage1_L2_cont2")->shape());
                        netSet.netB->blob_by_name("last_hm")->Reshape(netSet.netA->blob_by_name("Mconv7_stage2_L1_cont2")->shape());
                        netSet.netB->blob_by_name("last_fm")->Reshape(netSet.netVGG->blob_by_name("conv4_4_CPM")->shape());
                        netSet.netB->Reshape();

                        netSet.pafMem->Reshape(netSet.netB->blob_by_name("last_paf")->shape());
                        netSet.hmMem->Reshape(netSet.netB->blob_by_name("last_hm")->shape());
                        netSet.fmMem->Reshape(netSet.netB->blob_by_name("last_fm")->shape());
                        // netA PAF - Mconv10_stage1_L2_cont2
                        // netA HM - Mconv7_stage2_L1_cont2
                        // net B PAF - Mconv7_stage3_L2_cont2
                        // netB HM - Mconv7_stage4_L1_cont2

                        // Forward Net
                        memcpy(netSet.netVGG->blob_by_name("image")->mutable_cpu_data(), imageForNet.getConstPtr(), imageForNet.getVolume()*sizeof(float));
                        //op::uCharCvMatToFloatPtr(netSet.netVGG->blob_by_name("image")->mutable_cpu_data(), imageForNet, 1);
                        netSet.netVGG->Forward();
                        gpu_copy(netSet.netA->blob_by_name("conv4_4_CPM"), netSet.netVGG->blob_by_name("conv4_4_CPM"));
                        netSet.netA->Forward();

                        // Copy Mem
                        gpu_copy(netSet.pafMem, netSet.netA->blob_by_name("Mconv10_stage1_L2_cont2"));
                        gpu_copy(netSet.hmMem, netSet.netA->blob_by_name("Mconv7_stage2_L1_cont2"));
                        gpu_copy(netSet.fmMem, netSet.netVGG->blob_by_name("conv4_4_CPM"));

                        caffeNetOutputBlob.emplace_back(netSet.netA->blob_by_name("net_output"));

                    }else{

                        for(int s=0; s<1;s++){
                            // Copy state
                            gpu_copy(netSet.netB->blob_by_name("last_paf"), netSet.pafMem);
                            gpu_copy(netSet.netB->blob_by_name("last_hm"), netSet.hmMem);
                            gpu_copy(netSet.netB->blob_by_name("last_fm"), netSet.fmMem);

                            // Forward Net
                            memcpy(netSet.netVGG->blob_by_name("image")->mutable_cpu_data(), imageForNet.getConstPtr(), imageForNet.getVolume()*sizeof(float));
                            //op::uCharCvMatToFloatPtr(netSet.netVGG->blob_by_name("image")->mutable_cpu_data(), imageForNet, 1);
                            netSet.netVGG->Forward();
                            gpu_copy(netSet.netB->blob_by_name("conv4_4_CPM"), netSet.netVGG->blob_by_name("conv4_4_CPM"));
                            netSet.netB->Forward();

                            // Copy Mem
                            gpu_copy(netSet.pafMem, netSet.netB->blob_by_name("Mconv7_stage3_L2_cont2"));
                            gpu_copy(netSet.hmMem, netSet.netB->blob_by_name("Mconv7_stage4_L1_cont2"));
                            gpu_copy(netSet.fmMem, netSet.netVGG->blob_by_name("conv4_4_CPM"));
                        }
                        caffeNetOutputBlob.emplace_back(netSet.netB->blob_by_name("net_output"));

                        //visualize_hm(imageForNet,netSet.netB->blob_by_name("Mconv7_stage4_L1_cont2")->mutable_cpu_data(),netSet.netB->blob_by_name("Mconv7_stage4_L1_cont2")->shape(), 0, 21);
                    }

                }

                const auto caffeNetOutputBlobs = caffeNetSharedToPtr(caffeNetOutputBlob);

                if(upImpl->first_frame){
                    log("First Frame Reshape 2");
                    upImpl->spResizeAndMergeCaffe->Reshape(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()},
                                                 op::getPoseNetDecreaseFactor(upImpl->mPoseModel), 1.f/1.f, true,
                                                 0);
                    upImpl->spNmsCaffe->Reshape({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()}, op::getPoseMaxPeaks(),
                                      op::getPoseNumberBodyParts(upImpl->mPoseModel), 0);
                    upImpl->spBodyPartConnectorCaffe->Reshape({upImpl->spHeatMapsBlob.get(), upImpl->spPeaksBlob.get()});

                    const op::Point<int> imageSize{inputDataSize.x, inputDataSize.y};
                    //std::vector<double> scaleInputToNetInputs;
                    std::vector<op::Point<int>> netInputSizes;
                    double scaleInputToOutput;
                    op::Point<int> outputResolution;

                    mNetOutputSize = Point<int>{upImpl->mNetInput4DSizes[0][3],
                                                upImpl->mNetInput4DSizes[0][2]};

                    const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
                    upImpl->spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
                    const auto nmsThreshold = (float)get(PoseProperty::NMSThreshold);
                    upImpl->spNmsCaffe->setThreshold(nmsThreshold);


                    const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
                    const Point<int> netSize{intRound(scaleProducerToNetInput*inputDataSize.x),
                                             intRound(scaleProducerToNetInput*inputDataSize.y)};
                    mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, inputDataSize)};

                    upImpl->spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
                    upImpl->spBodyPartConnectorCaffe->setInterMinAboveThreshold(
                        (float)get(PoseProperty::ConnectInterMinAboveThreshold));
                    upImpl->spBodyPartConnectorCaffe->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
                    upImpl->spBodyPartConnectorCaffe->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
                    upImpl->spBodyPartConnectorCaffe->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));
                }

                // Process
                std::vector<caffe::Blob<float>*> heatMapsBlobs{upImpl->spHeatMapsBlob.get()};
                std::vector<caffe::Blob<float>*> peaksBlobs{upImpl->spPeaksBlob.get()};
                upImpl->spResizeAndMergeCaffe->Forward_gpu(caffeNetOutputBlobs, heatMapsBlobs); // ~5ms
                upImpl->spNmsCaffe->Forward_gpu(heatMapsBlobs, peaksBlobs);// ~2ms
                upImpl->spBodyPartConnectorCaffe->Forward_gpu({upImpl->spHeatMapsBlob.get(),upImpl->spPeaksBlob.get()},mPoseKeypoints, mPoseScores);


//                //cudaDeviceSynchronize();
//                std::cout << upImpl->spHeatMapsBlob.get()->shape_string() << std::endl;
//                std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
//                float time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.;
//                std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. <<std::endl;

//std::cout << mPoseKeypoints << std::endl;

                upImpl->tracker.run(mPoseKeypoints, upImpl->spHeatMapsBlob, upImpl->spPeaksBlob, 1./mScaleNetToOutput);



                // Set IDS
                std::vector<long long> ids;
                for (auto& kv : upImpl->tracker.tracklets_internal) {
                    ids.emplace_back(kv.first);
                }
                mPoseIds.reset(ids.size());
                for(int i=0; i<ids.size(); i++){
                    mPoseIds.at(i) = ids[i];
                }

                // Set Poses
                op::Array<float> tracklet_keypoints({(int)upImpl->tracker.tracklets_internal.size(), 21, 3},0.0f);
                int i=0;
                for (auto& kv : upImpl->tracker.tracklets_internal) {
                    for(int j=0; j<mPoseKeypoints.getSize(1); j++)
                        for(int k=0; k<mPoseKeypoints.getSize(2); k++)
                            tracklet_keypoints.at({i,j,k}) = kv.second.kp.at({j,k});
                    i+=1;
                }
                mPoseKeypoints = tracklet_keypoints.clone();
                // Scale Up
                for(int i=0; i<mPoseKeypoints.getSize()[0]; i++){
                    op::Array<float> person_kp = get_person_no_copy(mPoseKeypoints, i);
                    rescale_kp(person_kp, mScaleNetToOutput);
                }


                if(upImpl->first_frame) upImpl->first_frame = false;



//                for(int i=0; i<inputNetData.size(); i++){
//                    std::cout << inputNetData[i].printSize() << std::endl;
//                    std::cout << scaleInputToNetInputs[i] << std::endl;
//                }
//                std::cout << inputDataSize << std::endl;

//                std::cout << "---" << std::endl;


//                while (upImpl->spNets.size() < numberScales)
//                    addCaffeNetOnThread(upImpl->spNets, upImpl->spCaffeNetOutputBlobs, upImpl->mPoseModel,
//                                        upImpl->mGpuId, upImpl->mModelFolder, false);



//                // Process each image
//                for (auto i = 0u ; i < inputNetData.size(); i++)
//                {
//                    // 1. Caffe deep network
//                    // ~80ms
//                    upImpl->spNets.at(i)->forwardPass(inputNetData[i]);

//                    // Reshape blobs if required
//                    // Note: In order to resize to input size to have same results as Matlab, uncomment the commented
//                    // lines
//                    // Note: For dynamic sizes (e.g., a folder with images of different aspect ratio)
//                    const auto changedVectors = !vectorsAreEqual(
//                        upImpl->mNetInput4DSizes.at(i), inputNetData[i].getSize());
//                    if (changedVectors)
//                        // || !vectorsAreEqual(upImpl->mScaleInputToNetInputs, scaleInputToNetInputs))
//                    {
//                        upImpl->mNetInput4DSizes.at(i) = inputNetData[i].getSize();
//                        // upImpl->mScaleInputToNetInputs = scaleInputToNetInputs;
//                        reshapePoseExtractorCaffe(upImpl->spResizeAndMergeCaffe, upImpl->spNmsCaffe,
//                                                  upImpl->spBodyPartConnectorCaffe, upImpl->spMaximumCaffe,
//                                                  upImpl->spCaffeNetOutputBlobs, upImpl->spHeatMapsBlob,
//                                                  upImpl->spPeaksBlob, upImpl->spMaximumPeaksBlob,
//                                                  1.f, upImpl->mPoseModel, upImpl->mGpuId);
//                                                  // scaleInputToNetInputs[i] vs. 1.f
//                    }
//                    // Get scale net to output (i.e., image input)
//                    if (changedVectors || TOP_DOWN_REFINEMENT)
//                        mNetOutputSize = Point<int>{upImpl->mNetInput4DSizes[0][3],
//                                                    upImpl->mNetInput4DSizes[0][2]};
//                }
//                // 2. Resize heat maps + merge different scales
//                // ~5ms (GPU) / ~20ms (CPU)
//                const auto caffeNetOutputBlobs = caffeNetSharedToPtr(upImpl->spCaffeNetOutputBlobs);
//                const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
//                upImpl->spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
//                upImpl->spResizeAndMergeCaffe->Forward(caffeNetOutputBlobs, {upImpl->spHeatMapsBlob.get()});
//                // Get scale net to output (i.e., image input)
//                // Note: In order to resize to input size, (un)comment the following lines
//                const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
//                const Point<int> netSize{intRound(scaleProducerToNetInput*inputDataSize.x),
//                                         intRound(scaleProducerToNetInput*inputDataSize.y)};
//                mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, inputDataSize)};
//                // mScaleNetToOutput = 1.f;
//                // 3. Get peaks by Non-Maximum Suppression
//                // ~2ms (GPU) / ~7ms (CPU)
//                const auto nmsThreshold = (float)get(PoseProperty::NMSThreshold);
//                upImpl->spNmsCaffe->setThreshold(nmsThreshold);
//                const auto nmsOffset = float(0.5/double(mScaleNetToOutput));
//                upImpl->spNmsCaffe->setOffset(Point<float>{nmsOffset, nmsOffset});
//                upImpl->spNmsCaffe->Forward({upImpl->spHeatMapsBlob.get()}, {upImpl->spPeaksBlob.get()});
//                // 4. Connecting body parts
//                upImpl->spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
//                upImpl->spBodyPartConnectorCaffe->setInterMinAboveThreshold(
//                    (float)get(PoseProperty::ConnectInterMinAboveThreshold));
//                upImpl->spBodyPartConnectorCaffe->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
//                upImpl->spBodyPartConnectorCaffe->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
//                upImpl->spBodyPartConnectorCaffe->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));
//                // Note: BODY_25D will crash (only implemented for CPU version)
//                upImpl->spBodyPartConnectorCaffe->Forward(
//                    {upImpl->spHeatMapsBlob.get(), upImpl->spPeaksBlob.get()}, mPoseKeypoints, mPoseScores);
//                // Re-run on each person

//                // TODO: temporary code to test this works
//                mPoseIds.reset(mPoseScores.getVolume(), 99);
//                //log(mPoseIds);

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

    const float* PoseExtractorCaffe::getCandidatesCpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return upImpl->spPeaksBlob->cpu_data();
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

    const float* PoseExtractorCaffe::getCandidatesGpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return upImpl->spPeaksBlob->gpu_data();
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

    const float* PoseExtractorCaffe::getHeatMapCpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return upImpl->spHeatMapsBlob->cpu_data();
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

    const float* PoseExtractorCaffe::getHeatMapGpuConstPtr() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return upImpl->spHeatMapsBlob->gpu_data();
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

    std::vector<int> PoseExtractorCaffe::getHeatMapSize() const
    {
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return upImpl->spHeatMapsBlob->shape();
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

    const float* PoseExtractorCaffe::getPoseGpuConstPtr() const
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
