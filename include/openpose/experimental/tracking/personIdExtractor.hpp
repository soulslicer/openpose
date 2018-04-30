#ifndef OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
#define OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP

#include <unordered_map>
#include <openpose/core/common.hpp>
#include <openpose/experimental/tracking/personIdExtractor.hpp>
#include <opencv2/opencv.hpp>

namespace op
{
    struct PersonEntry
    {
        long long counterLastDetection;
        std::vector<cv::Point2f> keypoints;
        std::vector<cv::Point2f> lastKeypoints;
        std::vector<char> status;
        std::vector<cv::Point2f> getPredicted(){
            std::vector<cv::Point2f> predictedKeypoints(keypoints);
            if(!lastKeypoints.size()) return predictedKeypoints;
            for(size_t i=0; i<keypoints.size(); i++){
                predictedKeypoints[i] = cv::Point(predictedKeypoints[i].x + (keypoints[i].x-lastKeypoints[i].x),
                                                  predictedKeypoints[i].y + (keypoints[i].y-lastKeypoints[i].y));
            }
            return predictedKeypoints;
        }
        //std::vector<std::vector<cv::Point2f>> lastKeypoints;
        /*
        PersonEntry(long long _last_frame,
                    std::vector<cv::Point2f> _keypoints,
                    std::vector<char> _active):
                    last_frame(_last_frame), keypoints(_keypoints),
                    active(_active)
                    {}
        */
    };
    class OP_API PersonIdExtractor
    {

    public:
        PersonIdExtractor(const float confidenceThreshold = 0.1f, const float inlierRatioThreshold = 0.5f,
                          const float distanceThreshold = 30.f, const int numberFramesToDeletePerson = 10,
                          const int levels = 3, const int patchSize = 21, const bool trackVelocity = false);

        virtual ~PersonIdExtractor();

        Array<long long> extractIds(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput);
        void update(const cv::Mat& cvMatInput);
        Array<float> personEntriesAsOPArray();

        // Debug Params
        cv::Mat debugImage;
        void vizPersonEntries();
        void drawIDs(cv::Mat& img);

    private:
        const int mLevels;
        const int mPatchSize;
        const float mConfidenceThreshold;
        const float mInlierRatioThreshold;
        const float mDistanceThreshold;
        const int mNumberFramesToDeletePerson;
        const bool mTrackVelocity;
        long long mNextPersonId;
        cv::Mat mImagePrevious;
        std::vector<cv::Mat> mPyramidImagesPrevious;
        std::unordered_map<int, PersonEntry> mPersonEntries;
        DELETE_COPY(PersonIdExtractor);
    };
}

#endif // OPENPOSE_TRACKING_PERSON_ID_EXTRACTOR_HPP
