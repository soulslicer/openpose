#include <openpose/experimental/tracking/pyramidalLK.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/experimental/tracking/personIdExtractor.hpp>

// #define LK_CUDA

namespace op
{
    float getEuclideanDistance(const cv::Point2f& a, const cv::Point2f& b)
    {
        try
        {
            const auto difference = a - b;
            return std::sqrt(difference.x * difference.x + difference.y * difference.y);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.f;
        }
    }

    std::vector<PersonEntry> captureKeypoints(const Array<float>& poseKeypoints, const float confidenceThreshold)
    {
        try
        {
            // Define result
            std::vector<PersonEntry> personEntries(poseKeypoints.getSize(0));
            // Fill personEntries
            for (auto p = 0; p < (int)personEntries.size(); p++)
            {
                // Create person entry in the tracking map
                auto& personEntry = personEntries[p];
                auto& keypoints = personEntry.keypoints;
                auto& status = personEntry.status;
                personEntry.counterLastDetection = 0;

                for (auto kp = 0; kp < poseKeypoints.getSize(1); kp++)
                {
                    cv::Point2f cp;
                    cp.x = poseKeypoints[{p,kp,0}];
                    cp.y = poseKeypoints[{p,kp,1}];
                    keypoints.emplace_back(cp);

                    if (poseKeypoints[{p,kp,2}] < confidenceThreshold)
                        status.emplace_back(1);
                    else
                        status.emplace_back(0);
                }
            }
            // Return result
            return personEntries;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    void updateLK(std::unordered_map<int,PersonEntry>& personEntries, std::vector<cv::Mat>& pyramidImagesPrevious,
                  std::vector<cv::Mat>& pyramidImagesCurrent, const cv::Mat& imagePrevious,
                  const cv::Mat& imageCurrent, const int numberFramesToDeletePerson,
                  const int levels, const int patchSize, const bool trackVelocity)
    {
        try
        {
            // Get all key values
            // Otherwise, `erase` provokes core dumped when removing elements
            std::vector<int> keyValues;
            keyValues.reserve(personEntries.size());
            for (const auto& entry : personEntries)
                keyValues.emplace_back(entry.first);
            // Update or remove elements
            for (auto& key : keyValues)
            {
                auto& element = personEntries[key];

                // Remove keypoint
                if (element.counterLastDetection++ > numberFramesToDeletePerson){
                    //std::cout << "Erasing: " << key << std::endl;
                    personEntries.erase(key);
                // Update all keypoints for that entry
                }else
                {
                    PersonEntry personEntry;
                    personEntry.counterLastDetection = element.counterLastDetection;
                    #ifdef LK_CUDA
                        UNUSED(pyramidImagesPrevious);
                        UNUSED(pyramidImagesCurrent);
                        pyramidalLKGpu(element.keypoints, personEntry.keypoints, element.status,
                                       imagePrevious, imageCurrent, 3, 21);
                    #else
                        if(trackVelocity)
                        {
                            personEntry.keypoints = element.getPredicted();
                            pyramidalLKOcv(element.keypoints, personEntry.keypoints, pyramidImagesPrevious,
                                           pyramidImagesCurrent, element.status, imagePrevious, imageCurrent, levels, patchSize, true);
                        }
                        else
                            pyramidalLKOcv(element.keypoints, personEntry.keypoints, pyramidImagesPrevious,
                                           pyramidImagesCurrent, element.status, imagePrevious, imageCurrent, levels, patchSize);
                    #endif
                    personEntry.status = element.status;
                    personEntry.lastKeypoints = element.keypoints;
                    element = personEntry;
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void initializeLK(std::unordered_map<int,PersonEntry>& personEntries,
                     long long& mNextPersonId,
                     const Array<float>& poseKeypoints,
                     const float confidenceThreshold)
    {
        try
        {
            for (auto p = 0; p < poseKeypoints.getSize(0); p++)
            {
                const int currentPerson = mNextPersonId++;

                // Create person entry in the tracking map
                auto& personEntry = personEntries[currentPerson];
                auto& keypoints = personEntry.keypoints;
                auto& status = personEntry.status;
                personEntry.counterLastDetection = 0;

                for (auto kp = 0; kp < poseKeypoints.getSize(1); kp++)
                {
                    const cv::Point2f cp{poseKeypoints[{p,kp,0}], poseKeypoints[{p,kp,1}]};
                    keypoints.emplace_back(cp);

                    if (poseKeypoints[{p,kp,2}] < confidenceThreshold)
                        status.emplace_back(1);
                    else
                        status.emplace_back(0);
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Array<long long> matchLKAndOP(std::unordered_map<int,PersonEntry>& personEntries,
                                  long long& nextPersonId,
                                  const std::vector<PersonEntry>& openposePersonEntries,
                                  const cv::Mat& imagePrevious,
                                  const float inlierRatioThreshold,
                                  const float distanceThreshold)
    {
        try
        {
            Array<long long> poseIds{(int)openposePersonEntries.size(), -1};
            std::unordered_map<int, PersonEntry> pendingQueue;

            if (!openposePersonEntries.empty())
            {
                const auto numberKeypoints = openposePersonEntries[0].keypoints.size();
                for (auto i = 0u; i < openposePersonEntries.size(); i++)
                {
                    auto& poseId = poseIds.at(i);
                    const auto& openposePersonEntry = openposePersonEntries.at(i);
                    const auto personDistanceThreshold = fastMax(10.f,
                        distanceThreshold*float(std::sqrt(imagePrevious.cols*imagePrevious.rows)) / 960.f);

                    // Find best correspondance in the LK set
                    auto bestMatch = -1ll;
                    auto bestScore = 0.f;
                    for (const auto& personEntry : personEntries)
                    {
                        const auto& element = personEntry.second;
                        auto inliers = 0;
                        auto active = 0;

                        // Security checks
                        if (element.status.size() != numberKeypoints)
                            error("element.status.size() != numberKeypoints ||", __LINE__, __FUNCTION__, __FILE__);
                        if (openposePersonEntry.status.size() != numberKeypoints)
                            error("openposePersonEntry.status.size() != numberKeypoints",
                                  __LINE__, __FUNCTION__, __FILE__);
                        if (element.keypoints.size() != numberKeypoints)
                            error("element.keypoints.size() != numberKeypoints ||", __LINE__, __FUNCTION__, __FILE__);
                        if (openposePersonEntry.keypoints.size() != numberKeypoints)
                            error("openposePersonEntry.keypoints.size() != numberKeypoints",
                                  __LINE__, __FUNCTION__, __FILE__);
                        // Iterate through all keypoints
                        for (auto kp = 0u; kp < numberKeypoints; kp++)
                        {
                            // If enough threshold
                            if (!element.status[kp] && !openposePersonEntry.status[kp])
                            {
                                active++;
                                const auto distance = getEuclideanDistance(element.keypoints[kp],
                                                                           openposePersonEntry.keypoints[kp]);
                                if (distance < personDistanceThreshold)
                                    inliers++;
                            }
                        }

                        if (active > 0)
                        {
                            const auto score = inliers / (float)active;
                            if (score > bestScore && score >= inlierRatioThreshold)
                            {
                                bestScore = score;
                                bestMatch = personEntry.first;
                            }
                        }
                    }
                    // Found a best match, update LK table and poseIds
                    if (bestMatch != -1)
                        poseId = bestMatch;
                    else
                        poseId = nextPersonId++;

                    // Pass last positions
                    auto personEntryCopy = openposePersonEntry;
                    if(bestMatch != -1){
                        personEntryCopy.lastKeypoints = personEntries[bestMatch].lastKeypoints;

                        // Keep last point
                        for(size_t x=0; x<personEntryCopy.keypoints.size(); x++){
                            const cv::Point& ik = personEntryCopy.keypoints[x];
                            const cv::Point& jk = personEntries[bestMatch].keypoints[x];
                            float distance = sqrt(pow(ik.x-jk.x,2)+pow(ik.y-jk.y,2));
                            if(distance < 5){
                                personEntryCopy.keypoints[x] = jk;
                            }else if(distance < 10){
                                personEntryCopy.keypoints[x] = cv::Point((jk.x+ik.x)/2.,(jk.y+ik.y)/2.);
                            }
                        }

                        // See if lost
                        for(size_t x=0; x<personEntries[bestMatch].keypoints.size(); x++){
                            if((int)personEntryCopy.status[x]){
                                personEntryCopy.keypoints[x] = personEntries[bestMatch].keypoints[x];
                            }
                        }

                        personEntryCopy.lastKeypoints = personEntryCopy.keypoints;
                    }

                    pendingQueue[poseId] = personEntryCopy;
                }
            }

            // Update LK table with pending queue
            for (auto& pendingQueueEntry: pendingQueue)
                personEntries[pendingQueueEntry.first] = pendingQueueEntry.second;

            return poseIds;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }

    op::Array<float> op::PersonIdExtractor::personEntriesAsOPArray()
    {
        op::Array<float> opArray;
        if(!mPersonEntries.size()) return opArray;
        int dims[] = { (int)mPersonEntries.size(), (int)mPersonEntries.begin()->second.keypoints.size(), 3 };
        cv::Mat opArrayMat(3,dims,CV_32FC1);
        int i=0;
        for (auto& kv : mPersonEntries) {
            const PersonEntry& pe = kv.second;
            for(int j=0; j<dims[1]; j++){
                opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 0) = pe.keypoints[j].x;
                opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 1) = pe.keypoints[j].y;
                opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 2) = !(int)pe.status[j];
                if(pe.keypoints[j].x == 0 && pe.keypoints[j].y == 0)
                    opArrayMat.at<float>(i*dims[1]*dims[2] + j*dims[2] + 2) = 0;
            }
            i++;
        }
        opArray.setFrom(opArrayMat);
        return opArray;
    }

    void op::PersonIdExtractor::update(const cv::Mat &cvMatInput)
    {
        try
        {
            if (mImagePrevious.empty())
            {
                throw std::runtime_error("Call extractIds first");
            }
            else
            {
                cv::Mat imageCurrent;
                std::vector<cv::Mat> pyramidImagesCurrent;
                cvMatInput.convertTo(imageCurrent, CV_32F);
                updateLK(mPersonEntries, mPyramidImagesPrevious, pyramidImagesCurrent, mImagePrevious, imageCurrent,
                         mNumberFramesToDeletePerson, mLevels, mPatchSize, mTrackVelocity);
                mImagePrevious = imageCurrent;
                mPyramidImagesPrevious = pyramidImagesCurrent;

                vizPersonEntries();
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    op::PersonIdExtractor::PersonIdExtractor(const float confidenceThreshold, const float inlierRatioThreshold,
                                             const float distanceThreshold, const int numberFramesToDeletePerson,
                                             const int levels, const int patchSize, const bool trackVelocity) :
        mConfidenceThreshold{confidenceThreshold},
        mInlierRatioThreshold{inlierRatioThreshold},
        mDistanceThreshold{distanceThreshold},
        mNumberFramesToDeletePerson{numberFramesToDeletePerson},
        mLevels{levels},
        mPatchSize{patchSize},
        mTrackVelocity{trackVelocity},
        mNextPersonId{0ll}
    {
        try
        {
            //error("PersonIdExtractor (`identification` flag) buggy and not working yet, but we are working on it!"
            //      " Coming soon!", __LINE__, __FUNCTION__, __FILE__);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PersonIdExtractor::~PersonIdExtractor()
    {
    }

    Array<long long> PersonIdExtractor::extractIds(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput)
    {
        try
        {
            Array<long long> poseIds;
            const auto openposePersonEntries = captureKeypoints(poseKeypoints, mConfidenceThreshold);
            // log(mPersonEntries.size());

            // First frame
            if (mImagePrevious.empty())
            {
                // Add first persons to the LK set
                initializeLK(mPersonEntries, mNextPersonId, poseKeypoints, mConfidenceThreshold);
                // Capture current frame as floating point
                cvMatInput.convertTo(mImagePrevious, CV_32F);
            }
            // Rest
            else
            {
                cv::Mat imageCurrent;
                std::vector<cv::Mat> pyramidImagesCurrent;
                cvMatInput.convertTo(imageCurrent, CV_32F);
                updateLK(mPersonEntries, mPyramidImagesPrevious, pyramidImagesCurrent, mImagePrevious, imageCurrent,
                                         mNumberFramesToDeletePerson, mLevels, mPatchSize, mTrackVelocity);
                mImagePrevious = imageCurrent;
                mPyramidImagesPrevious = pyramidImagesCurrent;
            }

            // Get poseIds and update LKset according to OpenPose set
            poseIds = matchLKAndOP(mPersonEntries, mNextPersonId, openposePersonEntries, mImagePrevious,
                                   mInlierRatioThreshold, mDistanceThreshold);

            vizPersonEntries();

            return poseIds;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }

    // Debug Functions

    void op::PersonIdExtractor::drawIDs(cv::Mat& img)
    {
        for (auto& kv : mPersonEntries) {
            const PersonEntry& pe = kv.second;
            cv::Point avg(0,0);
            int i=-1;
            int count = 0;
            for(cv::Point p : pe.keypoints){
                i++;
                if((p.x == 0 && p.y == 0) || (int)pe.status[i]) continue;
                avg.x += p.x;
                avg.y += p.y;
                count++;
            }
            if(!count) continue;
            avg.x /= count;
            avg.y /= count;
            cv::putText(img, std::to_string(kv.first), avg, cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255,255,255),1);
        }
    }

    void op::PersonIdExtractor::vizPersonEntries(){
        for ( auto& kv : mPersonEntries) {
            PersonEntry& pe = kv.second;
            std::vector<cv::Point2f> predictedKeypoints = pe.getPredicted();
            for(size_t i=0; i<pe.keypoints.size(); i++){
                cv::circle(debugImage, pe.keypoints[i], 3, cv::Scalar(255,0,0),CV_FILLED);
                cv::putText(debugImage, std::to_string(!(int)pe.status[i]), pe.keypoints[i], cv::FONT_HERSHEY_DUPLEX, 0.4, cv::Scalar(0,0,255),1);

                if(pe.lastKeypoints.size()){
                    cv::line(debugImage, pe.keypoints[i], pe.lastKeypoints[i],cv::Scalar(255,0,0));
                    cv::circle(debugImage, pe.lastKeypoints[i], 3, cv::Scalar(255,255,0),CV_FILLED);
                }
                if(predictedKeypoints.size() && mTrackVelocity){
                    cv::line(debugImage, pe.keypoints[i], predictedKeypoints[i],cv::Scalar(255,0,0));
                    cv::circle(debugImage, predictedKeypoints[i], 3, cv::Scalar(255,0,255),CV_FILLED);
                }
            }
        }
    }

}
