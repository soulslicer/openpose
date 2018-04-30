// ------------------------- OpenPose Library Tutorial - Pose - Example 2 - Extract Pose or Heatmap from Image -------------------------
// This second example shows the user how to:
// 1. Load an image (`filestream` module)
// 2. Extract the pose of that image (`pose` module)
// 3. Render the pose or heatmap on a resized copy of the input image (`pose` module)
// 4. Display the rendered pose or heatmap (`gui` module)
// In addition to the previous OpenPose modules, we also need to use:
// 1. `core` module: for the Array<float> class that the `pose` module needs
// 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively

// 3rdparty dependencies
// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif
// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <openpose/experimental/headers.hpp>
#include <dirent.h>
#include <openpose/experimental/tracking/mosse.hpp>

std::vector<std::string> getFiles(std::string folder){
    std::vector<std::string> files;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (folder.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            files.push_back(folder + ent->d_name);
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
        throw;
    }

    std::vector<std::string> filesNew;
    std::sort(files.begin(), files.end());
    for(std::string file : files){
        if(file == folder+"." || file == folder+"..") continue;
        filesNew.push_back(file);
    }
    return filesNew;
}

// See all the available parameter options withe the `--help` flag. E.g. `build/examples/openpose/openpose.bin --help`
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging/Other
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(video,                    "",             "Process the video.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution,           "-1x368",       "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
                                                        " decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
                                                        " closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
                                                        " any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
                                                        " input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
                                                        " e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_string(output_resolution,        "-1x-1",        "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " input image resolution.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_int32(part_to_show,              19,             "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
                                                        " part heat map, 19 for the background heat map, 20 for all the body part heat maps"
                                                        " together, 21 for all the PAFs, 22-40 for each body part pair PAF.");
DEFINE_bool(disable_blending,           false,          "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
                                                        " background, instead of being rendered into the original image. Related: `part_to_show`,"
                                                        " `alpha_pose`, and `alpha_pose`.");
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
                                                        " heatmap, 0 will only show the frame. Only valid for GPU rendering.");
DEFINE_string(image_dir,                "",             "Process a directory of images. Use `examples/media/` for our default example folder with 20"
                                                        " images. Read all standard formats (jpg, png, bmp, etc.).");
#include <random>
#include <complex>



bool start = false;
bool down = false;
cv::Point minp(274,69);
cv::Point maxp(309,113);
void my_mouse_callback( int event, int x, int y, int flags, void* param ) {
    if(event==CV_EVENT_LBUTTONDOWN){
        down = true;
        minp = cv::Point(x,y);
        maxp = cv::Point(x,y);
    }
    if(event==CV_EVENT_LBUTTONUP){
        start = true;
        maxp = cv::Point(x,y);
    }
    if(event==CV_EVENT_MOUSEMOVE){
        if(start) return;
        if(!down) return;
        maxp = cv::Point(x,y);
    }
}

int openPoseTutorialPose2()
{
    cv::VideoCapture capture;
    std::vector<std::string> imagePaths;
    if(FLAGS_video.size()){
        capture = cv::VideoCapture(FLAGS_video);
        if( !capture.isOpened() )
            throw "Error when reading steam_avi";
    }else if(FLAGS_image_dir.size()){
        imagePaths = getFiles(FLAGS_image_dir);
    }else{
        capture = cv::VideoCapture(0);
    }

    cv::namedWindow("win");
    cv::setMouseCallback("win",my_mouse_callback);

    std::shared_ptr<op::MOSSE> mossePtr;

    int i=10000;
    int fcount=-1;
    for( ; ; )
    {
        fcount++;

        cv::Mat frame;
        if(FLAGS_image_dir.size()){
            std::cout << imagePaths[fcount] << std::endl;
            frame = cv::imread(imagePaths[fcount]);
            if(frame.empty())
                break;
        }else{
            capture >> frame;
            if(frame.empty())
                break;
        }

        while(!start){
            cv::Mat frameclone = frame.clone();
            cv::rectangle(frameclone, minp, maxp, cv::Scalar(255,0,0),2);
            cv::imshow("win",frameclone);
            cv::waitKey(15);
        }

        // First frame
        cv::Mat gray;
        cv::cvtColor(frame, gray, CV_BGR2GRAY);
        if(fcount == 1){
            mossePtr = std::shared_ptr<op::MOSSE>(new op::MOSSE(gray, {minp.x,minp.y,maxp.x,maxp.y}));
        }else if(fcount == 0) continue;
        else{
            auto start = std::chrono::system_clock::now();
            mossePtr->update(gray);
            auto end = std::chrono::system_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
            mossePtr->draw_state(frame);
        }

        // Step 7 - Show Result
        cv::rectangle(frame, minp, maxp, cv::Scalar(255,0,0),2);
        cv::imshow("win",frame);
        int key = cv::waitKey(15);
        if( key == 32 ){
            while(1){
                key = cv::waitKey(15);
                if(key == 32) break;
            }
        }
    }

    op::log("Example 3 successfully finished.", op::Priority::High);
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseTutorialPose2
    return openPoseTutorialPose2();
}
