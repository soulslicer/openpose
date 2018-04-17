#ifndef OPENPOSE_PYTHON_HPP
#define OPENPOSE_PYTHON_HPP

// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <caffe/caffe.hpp>
#include <stdlib.h>

#define default_logging_level 3
#define default_output_resolution "-1x-1"
#define default_net_resolution "-1x368"
#define default_model_pose "COCO"
#define default_alpha_pose 0.6
#define default_scale_gap 0.3
#define default_scale_number 1
#define default_render_threshold 0.05
#define default_num_gpu_start 0
#define default_disable_blending false
#define default_model_folder "/home/raaj/openpose/models/"

class OpenPose{
public:
    std::unique_ptr<op::PoseExtractorCaffe> poseExtractorCaffe;
    std::unique_ptr<op::PoseCpuRenderer> poseRenderer;
    std::unique_ptr<op::FrameDisplayer> frameDisplayer;
    std::unique_ptr<op::ScaleAndSizeExtractor> scaleAndSizeExtractor;

    OpenPose(int FLAGS_logging_level = default_logging_level,
             std::string FLAGS_output_resolution = default_output_resolution,
             std::string FLAGS_net_resolution = default_net_resolution,
             std::string FLAGS_model_pose = default_model_pose,
             float FLAGS_alpha_pose = default_alpha_pose,
             float FLAGS_scale_gap = default_scale_gap,
             int FLAGS_scale_number = default_scale_number,
             float FLAGS_render_threshold = default_render_threshold,
             int FLAGS_num_gpu_start = default_num_gpu_start,
             int FLAGS_disable_blending = default_disable_blending,
             std::string FLAGS_model_folder = default_model_folder
             ){
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        caffe::Caffe::SetDevice(0);

        op::log("OpenPose Library Python Wrapper", op::Priority::High);
        // ------------------------- INITIALIZATION -------------------------
        // Step 1 - Set logging level
            // - 0 will output all the logging messages
            // - 255 will output nothing
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
        scaleAndSizeExtractor = std::unique_ptr<op::ScaleAndSizeExtractor>(new op::ScaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap));

        poseExtractorCaffe = std::unique_ptr<op::PoseExtractorCaffe>(new op::PoseExtractorCaffe{poseModel, FLAGS_model_folder, FLAGS_num_gpu_start});

        poseRenderer = std::unique_ptr<op::PoseCpuRenderer>(new op::PoseCpuRenderer{poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
                                                                                                    (float)FLAGS_alpha_pose});
        frameDisplayer = std::unique_ptr<op::FrameDisplayer>(new op::FrameDisplayer{"OpenPose Tutorial - Example 1", outputSize});


        // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
        poseExtractorCaffe->initializationOnThread();
        poseRenderer->initializationOnThread();
    }

    void forward(const cv::Mat& inputImage, op::Array<float>& poseKeypoints, cv::Mat& displayImage, bool display = false){
        op::OpOutputToCvMat opOutputToCvMat;
        op::CvMatToOpInput cvMatToOpInput;
        op::CvMatToOpOutput cvMatToOpOutput;
        if(inputImage.empty())
            op::error("Could not open or find the image: ", __LINE__, __FUNCTION__, __FILE__);
        const op::Point<int> imageSize{inputImage.cols, inputImage.rows};
        // Step 2 - Get desired scale sizes
        std::vector<double> scaleInputToNetInputs;
        std::vector<op::Point<int>> netInputSizes;
        double scaleInputToOutput;
        op::Point<int> outputResolution;
        std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
            = scaleAndSizeExtractor->extract(imageSize);
        // Step 3 - Format input image to OpenPose input and output formats
        const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
        // Step 4 - Estimate poseKeypoints
        poseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
        poseKeypoints = poseExtractorCaffe->getPoseKeypoints();

        if(display){
            auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
            // Step 5 - Render poseKeypoints
            poseRenderer->renderPose(outputArray, poseKeypoints, scaleInputToOutput);
            // Step 6 - OpenPose output format to cv::Mat
            displayImage = opOutputToCvMat.formatToCvMat(outputArray);

        }
    }
};

#ifdef __cplusplus
extern "C" {
#endif

typedef void* c_OP;
op::Array<float> output;

c_OP newOP(int logging_level,
           char* output_resolution,
           char* net_resolution,
           char* model_pose,
           float alpha_pose,
           float scale_gap,
           int scale_number,
           float render_threshold,
           int num_gpu_start,
           bool disable_blending,
           char* model_folder
           ){
    return new OpenPose(logging_level, output_resolution, net_resolution, model_pose, alpha_pose,
                        scale_gap, scale_number, render_threshold, num_gpu_start, disable_blending, model_folder);
}
void delOP(c_OP op){
    delete (OpenPose *)op;
}
void forward(c_OP op, unsigned char* img, size_t rows, size_t cols, int* size, unsigned char* displayImg, bool display){
    OpenPose* openPose = (OpenPose*)op;
    cv::Mat image(rows, cols, CV_8UC3, img);
    cv::Mat displayImage(rows, cols, CV_8UC3, displayImg);
    openPose->forward(image, output, displayImage, display);
    size[0] = output.getSize()[0];
    size[1] = output.getSize()[1];
    size[2] = output.getSize()[2];
    if(display)
    memcpy(displayImg, displayImage.ptr(), sizeof(unsigned char)*rows*cols*3);
}
void getOutputs(c_OP op, float* array){
    memcpy(array, output.getPtr(), output.getSize()[0]*output.getSize()[1]*output.getSize()[2]*sizeof(float));
}

#ifdef __cplusplus
}
#endif

#endif
