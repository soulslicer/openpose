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
int openPoseTutorialPose2()
{
    op::log("OpenPose Library Tutorial - Example 2.", op::Priority::High);
    // ------------------------- INITIALIZATION -------------------------
    // Step 1 - Set logging level
    // - 0 will output all the logging messages
    // - 255 will output nothing
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
              __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // Step 2 - Read Google flags (user defined configuration)
    // outputSize
    const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
    // netInputSize
    const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
    // poseModel
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    // Check no contradictory flags enabled
    if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
        op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
    if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
        op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
                  __LINE__, __FUNCTION__, __FILE__);
    // Logging
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // Step 3 - Initialize all required classes
    op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);
    op::CvMatToOpInput cvMatToOpInput{poseModel};
    op::CvMatToOpOutput cvMatToOpOutput;
    auto poseExtractorPtr = std::make_shared<op::PoseExtractorCaffe>(poseModel, FLAGS_model_folder,
                                                                     FLAGS_num_gpu_start);
    op::PoseGpuRenderer poseGpuRenderer{poseModel, poseExtractorPtr, (float)FLAGS_render_threshold,
                !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap};
    poseGpuRenderer.setElementToRender(FLAGS_part_to_show);
    op::PoseCpuRenderer poseRenderer{poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
                (float)FLAGS_alpha_pose};
    op::OpOutputToCvMat opOutputToCvMat;
    op::FrameDisplayer frameDisplayer{"OpenPose Tutorial - Example 2", outputSize};
    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    poseExtractorPtr->initializationOnThread();
    poseGpuRenderer.initializationOnThread();

    // Person ID
    op::PersonIdExtractor personIDExtractor(0.1, 0.5, 30, 10, 3, 21, true);

    // ------------------------- POSE ESTIMATION AND RENDERING -------------------------
    // Step 1 - Read and load image, error if empty (possibly wrong path)
    // Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);

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

        const op::Point<int> imageSize{frame.cols, frame.rows};
        // Step 2 - Get desired scale sizes
        std::vector<double> scaleInputToNetInputs;
        std::vector<op::Point<int>> netInputSizes;
        double scaleInputToOutput;
        op::Point<int> outputResolution;
        std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
                = scaleAndSizeExtractor.extract(imageSize);
        // Step 3 - Format input image to OpenPose input and output formats
        const auto netInputArray = cvMatToOpInput.createArray(frame, scaleInputToNetInputs, netInputSizes);
        auto outputArray = cvMatToOpOutput.createArray(frame, scaleInputToOutput, outputResolution);
        cv::Mat outputImage = frame.clone();
        personIDExtractor.debugImage = outputImage.clone();

        i++;
        if(i > 5){
            // Step 4 - Estimate poseKeypoints
            poseExtractorPtr->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
            const auto poseKeypoints = poseExtractorPtr->getPoseKeypoints();
            const auto scaleNetToOutput = poseExtractorPtr->getScaleNetToOutput();
            // Step 5 - Render pose
            //poseGpuRenderer.renderPose(outputArray, poseKeypoints, scaleInputToOutput, scaleNetToOutput);
            //poseRenderer.renderPose(outputArray, poseKeypoints, scaleInputToOutput);
            // Step 6 - OpenPose output format to cv::Mat
            outputImage = opOutputToCvMat.formatToCvMat(outputArray);

            op::log("Extract");
            personIDExtractor.extractIds(poseKeypoints, frame);
            i=0;
        }else{
            op::log("Update");
            personIDExtractor.update(frame);

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Render Test
        const auto poseKeypointsTest = personIDExtractor.personEntriesAsOPArray();

        poseRenderer.renderPose(outputArray, poseKeypointsTest, scaleInputToOutput);
        outputImage = opOutputToCvMat.formatToCvMat(outputArray);

        personIDExtractor.drawIDs(outputImage);

        // Merge Image
        cv::Mat finalImage;
        cv::hconcat(outputImage, personIDExtractor.debugImage, finalImage);

        // Step 7 - Show Result
        cv::imshow("win",finalImage);
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
