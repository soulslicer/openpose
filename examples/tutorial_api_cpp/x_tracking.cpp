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
#include <dirent.h>

// Custom OpenPose flags
// Producer
DEFINE_string(image_path, "/home/raaj/video_datasets/posetrack_data/images/bonn/000017_bonn/",
              "Process an image. Read all standard formats (jpg, png, bmp, etc.).");

#define MODEL_PATH "/home/raaj/openpose/tracker/"
#define default_logging_level 3
#define default_output_resolution "-1x-1"
#define default_net_resolution 368
#define default_model_pose "BODY_21A"
#define default_alpha_pose 0.6
#define default_scale_gap 0.25
#define default_scale_number 1
#define default_render_threshold 0.05
#define default_num_gpu_start 0
#define default_disable_blending false

template<typename Dtype>
void matToCaffe(Dtype* caffeImg, const cv::Mat& imgAug){
    const int imageAugmentedArea = imgAug.rows * imgAug.cols;
    auto* uCharPtrCvMat = (unsigned char*)(imgAug.data);
    for (auto y = 0; y < imgAug.rows; y++)
    {
        const auto yOffset = y*imgAug.cols;
        for (auto x = 0; x < imgAug.cols; x++)
        {
            const auto xyOffset = yOffset + x;
            // const cv::Vec3b& bgr = imageAugmented.at<cv::Vec3b>(y, x);
            auto* bgr = &uCharPtrCvMat[3*xyOffset];
            caffeImg[xyOffset] = (bgr[0] - 128) / 256.0;
            caffeImg[xyOffset + imageAugmentedArea] = (bgr[1] - 128) / 256.0;
            caffeImg[xyOffset + 2*imageAugmentedArea] = (bgr[2] - 128) / 256.0;
        }
    }
}

struct NetSet{
    std::shared_ptr<caffe::Net<float> > netVGG, netA, netB;
    void load(){
        // Load nets
        netVGG.reset(new caffe::Net<float>(std::string(MODEL_PATH)+"vgg.prototxt", caffe::TEST));
        netA.reset(new caffe::Net<float>(std::string(MODEL_PATH)+"pose_deploy_1.prototxt", caffe::TEST));
        netB.reset(new caffe::Net<float>(std::string(MODEL_PATH)+"pose_deploy_2.prototxt", caffe::TEST));
        netVGG->CopyTrainedLayersFrom(std::string(MODEL_PATH)+"pose_iter_264000.caffemodel");
        netA->CopyTrainedLayersFrom(std::string(MODEL_PATH)+"pose_iter_264000.caffemodel");
        netB->CopyTrainedLayersFrom(std::string(MODEL_PATH)+"pose_iter_264000.caffemodel");
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

    TrackingNet(){
        // Caffe crap
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        caffe::Caffe::SetDevice(0);
        google::InitGoogleLogging("XXX");
        google::SetCommandLineOption("GLOG_minloglevel", "2");
        op::ConfigureLog::setPriorityThreshold((op::Priority)default_logging_level);

        // Setup scales
        for(int i=0; i<default_scale_number; i++){
            scales.emplace_back(1 - i*default_scale_gap);
        }

        // Setup net
        for(auto scale : scales){
            nets.emplace_back(NetSet());
            nets.back().load();
        }

    }
    ~TrackingNet(){

    }

    void reset(){

    }

    void run(const cv::Mat& image){

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        std::vector<caffe::Blob<float>> blobsForNet;
        std::vector<cv::Mat> imagesOrig;
        process_frames(image, blobsForNet, imagesOrig);

        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. <<std::endl;

        for(int i=0; i<scales.size(); i++){
            NetSet& netSet = nets[i];
            caffe::Blob<float>& blob = blobsForNet[i];
            //netSet.netVGG->blob_by_name("image");
        }

//        cv::imshow("win", image);
//        cv::waitKey(15);

    }

    void process_frames(const cv::Mat& frame, std::vector<caffe::Blob<float>>& blobsForNet, std::vector<cv::Mat>& imagesOrig){
        // 1 = width, 0 = height
        const float boxsize = default_net_resolution;
        std::vector<int> base_net_res;
        blobsForNet = std::vector<caffe::Blob<float>>(scales.size());
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

            caffe::Blob<float>& blobForNet = blobsForNet[i];
            blobForNet.Reshape({1, 3, imageForNet.size().height, imageForNet.size().width});
            //matToCaffe(blobForNet.mutable_cpu_data(), imageForNet);
            op::uCharCvMatToFloatPtr(blobForNet.mutable_cpu_data(), imageForNet, 1);
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

    // Load images from folder
    std::vector<std::string> imagePaths = filesFromFolder(FLAGS_image_path);
    for(auto imagePath : imagePaths){
        cv::Mat img = cv::imread(FLAGS_image_path+imagePath);

        t.run(img);
    }


    return 0;
    //    // Running tutorialApiCpp3
    //    return tutorialApiCpp3();
}
