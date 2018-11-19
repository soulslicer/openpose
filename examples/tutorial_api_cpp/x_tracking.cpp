// ----------------------- OpenPose C++ API Tutorial - Example 3 - Body from image configurable -----------------------
// It reads an image, process it, and displays it with the pose (and optionally hand and face) keypoints. In addition,
// it includes all the OpenPose configuration flags (enable/disable hand, face, output saving, etc.).

// Command-line user intraface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// OpenPose dependencies
#include <caffe/caffe.hpp>
#include <openpose/gpu/cuda.hpp>
#include <openpose/net/bodyPartConnectorBase.hpp>
#include <dirent.h>

#include <x_util.h>

#include <chrono>
#include <thread>

// Custom OpenPose flags
// Producer
//../../video_datasets/posetrack_data/images/bonn_5sec/022688_mpii/*.jpg
//DEFINE_string(image_path, "/home/raaj/Storage/video_datasets/posetrack_data/images/bonn/000017_bonn/",
//              "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
DEFINE_string(image_path, "/home/raaj/Storage/video_datasets/posetrack_data/images/bonn_mpii_test_v2_5sec/12834_mpii/",
              "Process an image. Read all standard formats (jpg, png, bmp, etc.).");

#define MODEL_PATH "/home/raaj/openpose/tracker/"
#define FLAGS_logging_level 3
#define FLAGS_output_resolution "-1x-1"
#define FLAGS_net_resolution "-1x368"
#define FLAGS_resolution 368
#define FLAGS_model_pose "BODY_21A"
#define FLAGS_alpha_pose 0.6
#define FLAGS_scale_gap 0.5
#define FLAGS_scale_number 1
#define FLAGS_render_threshold 0.05
#define FLAGS_num_gpu_start 0
#define FLAGS_disable_blending false

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

        //return final_idxs;

//        // Kp iterate on person
//        for(int j=0; j<person_kp.getSize()[0]; j++){
//            auto body_part = get_bp_no_copy(person_kp, j);
//            if(body_part[2] < render_threshold) continue;

//            // Find best match
//            int best_cost = 1000000;
//            int best_tid = -1;

//            // Iterate current tracklets
//            for ( auto &trackletPair : tracklets_internal ) {
//                Tracklet& tracklet = trackletPair.second;
//                int tid = trackletPair.first;
//                if(!tracklet.valid) continue;
//                float tracklet_cost = 0;

//                auto tracklet_body_part = get_bp_no_copy(tracklet.kp, j);
//                if(tracklet_body_part[2] < render_threshold) continue;
//                float l2Dist = l2(tracklet_body_part, body_part);
//                if(l2Dist > 10) continue;
//                tracklet_cost += l2Dist;

//                if(tracklet_cost < best_cost){
//                    best_cost = tracklet_cost;
//                    best_tid = tid;
//                }
//            }

//            // Set TID
//            if(best_tid >= 0) final_idxs[j]=best_tid;
//        }
//        //return final_idxs;

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

//        std::cout << pid << std::endl;
//        print_vector(final_idxs);


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

        //if(frame_count == 1) exit(-1);

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
        // Need to convert my tracklets into op::Array
        std::pair<op::Array<float>, std::map<int, int>> taf_scores = taf_kernel(pose_keypoints, heatMapsBlob);

        // Iterate Pose Keypoints (Global Score)
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

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

        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        float time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.;
        std::cout << "A = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. <<std::endl;


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
        }
        for(auto to_del : to_delete) tracklets_internal.erase(tracklets_internal.find(to_del));

        std::cout << frame_count << std::endl;
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

class TrackingNet{
public:
    //caffe::Net<float> netA, netB;
    //Nets nets;
    std::vector<NetSet> nets;
    std::vector<float> scales;
    bool first_frame = true;
    double scaleInputToOutput;
    float pointScale;
    op::Point<int> outputResolution;
    std::map<std::string, int> COCO21_MAPPING;
    std::vector<std::vector<int>> colors;

    std::unique_ptr<op::PoseCpuRenderer> poseRenderer;
    std::unique_ptr<op::PoseExtractorCaffe> poseExtractorCaffe;
    std::unique_ptr<op::ScaleAndSizeExtractor> scaleAndSizeExtractor;
    std::unique_ptr<op::ResizeAndMergeCaffe<float>> resizeAndMergeCaffe;
    std::unique_ptr<op::NmsCaffe<float>> nmsCaffe;
    std::unique_ptr<op::BodyPartConnectorCaffe<float>> bodyPartConnectorCaffe;
    op::PoseModel poseModel;
    std::shared_ptr<caffe::Blob<float>> heatMapsBlob;
    std::shared_ptr<caffe::Blob<float>> peaksBlob;

    Tracker tracker;

    TrackingNet(){
        // Caffe crap
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        caffe::Caffe::SetDevice(0);
        google::InitGoogleLogging("XXX");
        google::SetCommandLineOption("GLOG_minloglevel", "2");
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);

        // Setup scales
        for(int i=0; i<FLAGS_scale_number; i++){
            scales.emplace_back(1 - i*FLAGS_scale_gap);
        }

        // Setup net
        for(auto scale : scales){
            nets.emplace_back(NetSet());
            nets.back().load();
        }

        // Mapping
        COCO21_MAPPING = {
            {"NOSE", 0},
            {"NECK", 1},
            {"RSHOULDER", 2},
            {"RELBOW", 3},
            {"RWRIST", 4},
            {"LSHOULDER", 5},
            {"LELBOW", 6},
            {"LWRIST", 7},
            {"LOWERABS", 8},
            {"RHIP", 9},
            {"RKNEE", 10},
            {"RANKLE", 11},
            {"LHIP", 12},
            {"LKNEE", 13},
            {"LANKLE", 14},
            {"REYE", 15},
            {"LEYE", 16},
            {"REAR", 17},
            {"LEAR", 18},
            {"REALNECK", 19},
            {"TOP", 20},
        };
        colors = {{255,0,0},{0,255,0},{0,0,255},{255,255,0},{0,255,255},{255,0,255},{128,255,0},{0,128,255},{128,0,255},{128,0,0},{0,128,0},{0,0,128},{50,50,0},{50,50,0},{0,50,128},{0,100,128},{100,30,128},{0,50,255},{50,255,128},{100,90,128},{0,90,255},{50,90,128},{255,0,0},{0,255,0},{0,0,255},{255,255,0},{0,255,255},{255,0,255},{128,255,0},{0,128,255},{128,0,255},{128,0,0},{0,128,0},{0,0,128},{50,50,0},{50,50,0},{0,50,128},{0,100,128},{100,30,128},{0,50,255},{50,255,128},{100,90,128},{0,90,255},{50,90,128}};

        // Other
        poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        poseExtractorCaffe = std::unique_ptr<op::PoseExtractorCaffe>(new op::PoseExtractorCaffe{ poseModel, FLAGS_model_folder, FLAGS_num_gpu_start });
        scaleAndSizeExtractor = std::unique_ptr<op::ScaleAndSizeExtractor>(new op::ScaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap));
        resizeAndMergeCaffe = std::unique_ptr<op::ResizeAndMergeCaffe<float>>(new op::ResizeAndMergeCaffe<float>{});
        nmsCaffe = std::unique_ptr<op::NmsCaffe<float>>(new op::NmsCaffe<float>{});
        bodyPartConnectorCaffe = std::unique_ptr<op::BodyPartConnectorCaffe<float>>(new op::BodyPartConnectorCaffe<float>{});
        heatMapsBlob = { std::make_shared<caffe::Blob<float>>(1,1,1,1) };
        peaksBlob = { std::make_shared<caffe::Blob<float>>(1,1,1,1) };
        bodyPartConnectorCaffe->setPoseModel(poseModel);
        poseRenderer = std::unique_ptr<op::PoseCpuRenderer>(new op::PoseCpuRenderer{ poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
                                                                                     (float)FLAGS_alpha_pose });

    }
    //    ~TrackingNet(){

    //    }

    void reset(){
        first_frame = true;
        tracker.tracklets_internal.clear();
    }


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

    std::vector<caffe::Blob<float>*> caffeNetSharedToPtr(
            std::vector<boost::shared_ptr<caffe::Blob<float>>>& caffeNetOutputBlob)
    {
        try
        {
            // Prepare spCaffeNetOutputBlobss
            std::vector<caffe::Blob<float>*> caffeNetOutputBlobs(caffeNetOutputBlob.size());
            for (auto i = 0u; i < caffeNetOutputBlobs.size(); i++)
                caffeNetOutputBlobs[i] = caffeNetOutputBlob[i].get();
            return caffeNetOutputBlobs;
        }
        catch (const std::exception& e)
        {
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return{};
        }
    }

    void visualize_hm(cv::Mat& imageForNet, float* heatmaps, std::vector<int> size, int cstart, int cend){
        float netDecreaseFactor = (float)(imageForNet.size().height / size[2]);
        int num_maps = cend-cstart;
        std::vector<cv::Mat> resized_heatmaps(num_maps);
        cv::Mat accum = cv::Mat(size[2]*netDecreaseFactor, size[3]*netDecreaseFactor, CV_32FC1, cv::Scalar(0.));
        for(int i=cstart; i<cend; i++){
            cv::Mat& hm = resized_heatmaps[i-cstart];
            hm = cv::Mat(size[2], size[3], CV_32FC1, &heatmaps[i*size[2]*size[3]]);
            cv::resize(hm, hm, cv::Size(0,0), netDecreaseFactor, netDecreaseFactor);
            accum += hm;
        }
        accum = accum*255;
        accum.convertTo(accum, CV_8UC1);
        cv::applyColorMap(accum, accum, cv::COLORMAP_JET);
        cv::Mat final;
        cv::addWeighted(imageForNet, 0.5, accum, 0.5, 0, final);
        cv::imshow("win", final);
        cv::waitKey(15);

    }

    float run(const cv::Mat& image){


        //std::vector<caffe::Blob<float>> blobsForNet;
        std::vector<cv::Mat> imagesOrig;
        process_frames(image, imagesOrig);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


        std::vector<boost::shared_ptr<caffe::Blob<float>>> caffeNetOutputBlob;

        for(int i=0; i<scales.size(); i++){
            NetSet& netSet = nets[i];
            cv::Mat& imageForNet = imagesOrig[i];

            if(first_frame){

                // Do Reshaping once
                netSet.netVGG->blob_by_name("image")->Reshape({1, 3, imageForNet.size().height, imageForNet.size().width});
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
                op::uCharCvMatToFloatPtr(netSet.netVGG->blob_by_name("image")->mutable_cpu_data(), imageForNet, 1);
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
                    op::uCharCvMatToFloatPtr(netSet.netVGG->blob_by_name("image")->mutable_cpu_data(), imageForNet, 1);
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

        if(first_frame){
            resizeAndMergeCaffe->Reshape(caffeNetOutputBlobs, {heatMapsBlob.get()},
                                         op::getPoseNetDecreaseFactor(poseModel), 1.f/1.f, true,
                                         0);
            nmsCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()}, op::getPoseMaxPeaks(),
                              op::getPoseNumberBodyParts(poseModel), 0);
            bodyPartConnectorCaffe->Reshape({heatMapsBlob.get(), peaksBlob.get()});

            const op::Point<int> imageSize{image.cols, image.rows};
            std::vector<double> scaleInputToNetInputs;
            std::vector<op::Point<int>> netInputSizes;
            //double scaleInputToOutput;
            //op::Point<int> outputResolution;

            std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
                    = scaleAndSizeExtractor->extract(imageSize);
            const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
            resizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
            nmsCaffe->setThreshold((float)poseExtractorCaffe->get(op::PoseProperty::NMSThreshold));

            float mScaleNetToOutput = 1./scaleInputToNetInputs[0];
            pointScale = mScaleNetToOutput;
            bodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
            bodyPartConnectorCaffe->setInterMinAboveThreshold(
                        (float)poseExtractorCaffe->get(op::PoseProperty::ConnectInterMinAboveThreshold)
                        );
            bodyPartConnectorCaffe->setInterThreshold((float)poseExtractorCaffe->get(op::PoseProperty::ConnectInterThreshold));
            bodyPartConnectorCaffe->setMinSubsetCnt((int)poseExtractorCaffe->get(op::PoseProperty::ConnectMinSubsetCnt));
            bodyPartConnectorCaffe->setMinSubsetScore((float)poseExtractorCaffe->get(op::PoseProperty::ConnectMinSubsetScore));

        }

        // Process
        std::vector<caffe::Blob<float>*> heatMapsBlobs{heatMapsBlob.get()};
        std::vector<caffe::Blob<float>*> peaksBlobs{peaksBlob.get()};
        resizeAndMergeCaffe->Forward_gpu(caffeNetOutputBlobs, heatMapsBlobs); // ~5ms
        nmsCaffe->Forward_gpu(heatMapsBlobs, peaksBlobs);// ~2ms
        op::Array<float> mPoseKeypoints;
        op::Array<float> mPoseScores;
        bodyPartConnectorCaffe->Forward_gpu({heatMapsBlob.get(),peaksBlob.get()},mPoseKeypoints, mPoseScores);

        // My own draw function that takes in posekeypoints and mids for drawing

        // TAF Score calculator?

        //std::cout << mPoseKeypoints.printSize() << std::endl;
        //std::cout << peaksBlobs.back()->shape_string() << std::endl;


        tracker.run(mPoseKeypoints, heatMapsBlob, peaksBlob, 1./pointScale);

        cudaDeviceSynchronize();
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        float time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.;
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. <<std::endl;





        //visualize_hm(imagesOrig[0],heatMapsBlobs.back()->mutable_cpu_data(),heatMapsBlobs.back()->shape(), 0, 21);
        cv::Mat drawImage = image.clone();
        //cv::Mat otherImage = image.clone();
        for (auto& kv : tracker.tracklets_internal) {
            auto tidx = kv.first;
            auto& tracklet = kv.second;
            //cv::putText(drawImage,"P"+std::to_string(tidx), (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[tidx  % len(colors)], 2)

            cv::Point xx(tracklet.kp.at({0,0})*pointScale, tracklet.kp.at({0,1})*pointScale);
            cv::Scalar color(colors[tidx % colors.size()][0], colors[tidx % colors.size()][1], colors[tidx % colors.size()][2]);

            cv::putText(drawImage,"T"+std::to_string(tidx), xx, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);


            draw_lines_coco_21(drawImage, tracklet.kp, tidx, COCO21_MAPPING, colors, pointScale);
            //if(tracklet.kp_hitcount - frame_count < 0) to_delete.emplace_back(tidx);
        }



//        //std::cout << frame_
//        cv::putText(drawImage,"F"+std::to_string(tracker.frame_count), cv::Point(20,20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(100), 2);

        cv::imshow("win", drawImage);
//       // cv::imshow("other", otherImage);
        int key = cv::waitKey(2);
////        if(tracker.frame_count == 25){
////            while(1){
////            cv::imshow("win", drawImage);
////            cv::imshow("other", otherImage);
////            int key = cv::waitKey(15);
////            }
////        }
//        //std::this_thread::sleep_for(std::chrono::milliseconds(x));




        if(first_frame) first_frame = false;

        return time;

        //        cv::imshow("win", image);
        //        cv::waitKey(15);

    }

    void process_frames(const cv::Mat& frame, std::vector<cv::Mat>& imagesOrig){
        // 1 = width, 0 = height
        const float boxsize = FLAGS_resolution;
        std::vector<int> base_net_res;
        //blobsForNet = std::vector<caffe::Blob<float>>(scales.size());
        imagesOrig = std::vector<cv::Mat>(scales.size());

        for(int i=0; i<scales.size(); i++){
            float scale = scales[i];
            std::vector<int> net_res;
            if(i == 0){
                net_res = {16 * (int)((boxsize * frame.size().width / float(frame.size().height) / 16) + 0.5), (int)boxsize};
                base_net_res = net_res;
            }else{
                net_res = {(int)(std::min(base_net_res[0], std::max(1, (int)((base_net_res[0] * scale)+0.5)/16*16))),
                           (int)(std::min(base_net_res[1], std::max(1, (int)((base_net_res[1] * scale)+0.5)/16*16)))};
            }
            std::vector<int> input_res = {frame.size().width, frame.size().height};
            float scale_factor = std::min((net_res[0] - 1) / (float)(input_res[0] - 1), (net_res[1] - 1) / (float)(input_res[1] - 1));
            cv::Mat warp_matrix = (cv::Mat_<float>(2,3) << scale_factor, 0, 0,
                                   0, scale_factor, 0);
            cv::Mat imageForNet;
            int flag; if(scale_factor < 1.) flag = cv::INTER_AREA; else flag = cv::INTER_CUBIC;
            if(scale_factor != 1){
                cv::warpAffine(frame, imageForNet, warp_matrix, cv::Size(net_res[0], net_res[1]), flag, cv::BORDER_CONSTANT, {0,0,0});
            }else{
                imageForNet = frame.clone();
            }

            imagesOrig[i] = imageForNet;

            //            caffe::Blob<float>& blobForNet = blobsForNet[i];
            //            blobForNet.Reshape({1, 3, imageForNet.size().height, imageForNet.size().width});
            //            //matToCaffe(blobForNet.mutable_cpu_data(), imageForNet);
            //            op::uCharCvMatToFloatPtr(blobForNet.mutable_cpu_data(), imageForNet, 1);
        }

    }


};

std::vector<std::string> filesFromFolder(std::string folder){
    std::vector<std::string> files;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (folder.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            if(ent->d_name[0] == '.') continue;
            files.emplace_back(ent->d_name);
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
    }
    std::sort(files.begin(),files.end());
    return files;
}

int main(int argc, char *argv[])
{

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    TrackingNet t = TrackingNet();


    cv::VideoCapture cap;

    if(!cap.open(0))
        return 0;
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);

//    if(!cap.open("/home/raaj/Desktop/reid.mp4"))
//        return 0;

    int skip=0;

    for(;;)
    {


          cv::Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream

          skip++;
          if(skip < 3){
              continue;
          }
          skip = 0;

          t.run(frame);

    }
    // the camera will be closed automatically upon exit
    // cap.close();

//    // Load images from folder
//    std::vector<std::string> imagePaths = filesFromFolder(FLAGS_image_path);
//    for(auto imagePath : imagePaths){
//        cv::Mat img = cv::imread(FLAGS_image_path+imagePath);

//        t.run(img);
//    }




    return 0;
    //    // Running tutorialApiCpp3
    //    return tutorialApiCpp3();
}
