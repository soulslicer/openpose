// #include <iostream>
#include <opencv2/core/core.hpp> // cv::Point2f, cv::Mat
#include <opencv2/imgproc/imgproc.hpp> // cv::pyrDown
#include <openpose/experimental/tracking/mosse.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

namespace op
{
    MOSSE::MOSSE(cv::Mat frame, std::array<int, 4> rect)
    {
        // Get an ideal fft size window
        int x1 = rect[0]; int y1 = rect[1]; int x2 = rect[2]; int y2 = rect[3];
        int w = cv::getOptimalDFTSize(x2-x1);
        int h = cv::getOptimalDFTSize(y2-y1);
        x1 = (x1+x2-w)/2;
        y1 = (y1+y2-h)/2;

        // Get image patch from rect
        float x = x1+0.5*(w-1); float y = y1+0.5*(h-1);
        pos.x = x; pos.y = y;
        size.width = w; size.height = h;
        cv::Mat img; cv::getRectSubPix(frame,size,pos,img);

        // Get a gaussian kernel and a hanning window
        cv::createHanningWindow(win, size, CV_32F);
        cv::Mat g = cv::Mat(size,CV_32FC1,cv::Scalar(0.));
        g.at<float>(h/2,w/2) = 1;
        cv::GaussianBlur(g,g,cv::Size(-1,-1),2.0);
        double minVal; double maxVal; int minIdx, maxIdx;
        cv::minMaxIdx(g, &minVal, &maxVal, &minIdx, &maxIdx);
        g /= maxVal;

        // FT of gaussian patch
        cv::dft(g,G,cv::DFT_COMPLEX_OUTPUT);
        H1 = G.clone(); H1 = cv::Scalar::all(0.);
        H2 = G.clone(); H2 = cv::Scalar::all(0.);

        // Sample different warps and add spectrum
        for(int i=0; i<128; i++){
            cv::Mat a = preprocess(rnd_warp(img));
            cv::Mat A; cv::dft(a, A, cv::DFT_COMPLEX_OUTPUT);
            cv::Mat GA; cv::mulSpectrums(this->G, A, GA, 0, true);
            cv::Mat AA; cv::mulSpectrums(A, A, AA, 0, true);
            this->H1 += GA;
            this->H2 += AA;
        }

        update_kernel();
        update(frame);
    }

    void MOSSE::draw_state(cv::Mat &vis)
    {
        float x = this->pos.x; float y = this->pos.y; float w = this->size.width; float h = this->size.height;
        float x1 = (int)(x-0.5*w); float y1 = (int)(y-0.5*h); float x2 = (int)(x+0.5*w); float y2 = (int)(y+0.5*h);
        cv::rectangle(vis, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(0,0,255));
        if(this->good){
            cv::circle(vis, cv::Point(x,y), 2, cv::Scalar(0,0,255));
            cv::putText(vis, std::to_string(this->psr), cv::Point(x2,y2), CV_FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0,255,0));
        }else{
            cv::line(vis, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(0,0,255));
            cv::line(vis, cv::Point(x2,y1), cv::Point(x1,y2), cv::Scalar(0,0,255));
        }
    }

    void MOSSE::update(cv::Mat frame)
    {
        cv::Mat img;
        cv::getRectSubPix(frame, this->size, this->pos, img);
        this->last_img = img.clone();
        img = preprocess(img);
        cv::Point2f delta;
        std::tie(this->last_resp, delta, this->psr) = correlate(img);

        if(this->psr > min_psr) this->good = true;
        else{
            this->good = false;
            if(loss_action == 0){
                return;
            }else if(loss_action == 1){
                // FT of gaussian patch
                H1 = G.clone(); H1 = cv::Scalar::all(0.);
                H2 = G.clone(); H2 = cv::Scalar::all(0.);

                // Sample different warps and add spectrum
                for(int i=0; i<128; i++){
                    cv::Mat a = preprocess(rnd_warp(this->last_img));
                    cv::Mat A; cv::dft(a, A, cv::DFT_COMPLEX_OUTPUT);
                    cv::Mat GA; cv::mulSpectrums(this->G, A, GA, 0, true);
                    cv::Mat AA; cv::mulSpectrums(A, A, AA, 0, true);
                    this->H1 += GA;
                    this->H2 += AA;
                }

                update_kernel();
            }
        }

        this->pos = this->pos + delta;
        cv::getRectSubPix(frame, this->size, this->pos, img);
        this->last_img = img.clone();
        img = preprocess(img);

        cv::Mat A;
        cv::dft(img, A, cv::DFT_COMPLEX_OUTPUT);
        cv::Mat H1t, H2t;
        cv::mulSpectrums(this->G, A, H1t, 0, true);
        cv::mulSpectrums(A, A, H2t, 0, true);
        this->H1 = this->H1*(1.0-lr) + H1t*lr;
        this->H2 = this->H2*(1.0-lr) + H2t*lr;

        update_kernel();
    }

    std::tuple<cv::Mat, cv::Point2f, float> MOSSE::correlate(const cv::Mat &img)
    {
        cv::Mat imgDFT; cv::dft(img, imgDFT, cv::DFT_COMPLEX_OUTPUT);
        cv::Mat C; cv::mulSpectrums(imgDFT, this->H, C, 0, true);
        cv::Mat resp; cv::idft(C, resp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
        double minVal, maxVal; cv::Point minLoc, maxLoc;
        cv::minMaxLoc(resp, &minVal, &maxVal, &minLoc, &maxLoc);
        cv::Mat side_resp = resp.clone();
        cv::rectangle(side_resp, cv::Point(maxLoc.x-5, maxLoc.y-5), cv::Point(maxLoc.x+5, maxLoc.y+5), 0, -1);
        cv::Mat imgMean, imgStd;
        cv::meanStdDev(side_resp, imgMean, imgStd);
        float smean = imgMean.at<double>(0); float sstd = imgStd.at<double>(0);
        float psr = (maxVal-smean) / (sstd+1e-5);
        return std::tuple<cv::Mat, cv::Point2f, float>{resp, cv::Point2f(maxLoc.x-(int)(resp.size().width/2),maxLoc.y-(int)(resp.size().height/2)), psr};
    }

    void MOSSE::update_kernel()
    {
        this->H = divSpec(this->H1, this->H2);
        negImag(this->H);
        //cv::Mat splitted[2] = {cv::Mat(this->H.size(),CV_32F),cv::Mat(this->H.size(),CV_32F)};
        //cv::split(this->H, splitted);
        //cv::imwrite("/home/raaj/a.png", splitted[1]*255);
        //exit(-1);
    }

    cv::Mat MOSSE::divSpec(const cv::Mat &A, const cv::Mat &B)
    {
        // Inv B more efficient
        cv::Mat invB = B.clone();
        float* invBptr = &invB.at<float>(0,0);
        const float* Bptr = &B.at<float>(0,0);
        for(size_t i=0; i<B.total(); i++){
            float real = Bptr[i*2 + 0];
            float imag = Bptr[i*2 + 1];
            float square = real*real + imag*imag;
            invBptr[i*2 + 0] = (real/square);
            invBptr[i*2 + 1] = -(imag/square);
        }
        cv::Mat C; cv::mulSpectrums(A, invB, C, 0, true);
        return C;
    }

    void MOSSE::negImag(cv::Mat &B)
    {
        float* bPtr = &B.at<float>(0,0);
        for(size_t i=0; i<B.total(); i++){
            bPtr[i*2 + 1] = -bPtr[i*2 + 1];
        }
    }

    cv::Mat MOSSE::preprocess(const cv::Mat &img)
    {
        cv::Mat imgFloat;
        img.convertTo(imgFloat, CV_32F);
        cv::log(imgFloat + 1.0, imgFloat);
        cv::Mat imgMean, imgStd;
        cv::meanStdDev(imgFloat, imgMean, imgStd);
        float smean = imgMean.at<double>(0); float sstd = imgStd.at<double>(0);
        imgFloat = (imgFloat - smean) / (sstd + 1e-5);
        //imgFloat = (imgFloat - imgMean) / (imgStd + 1e-5);
        return imgFloat.mul(this->win);
    }

    cv::Mat MOSSE::rnd_warp(const cv::Mat &a)
    {
        // Create random warp matrix
        float h = a.size().height; float w = a.size().width;
        cv::Mat T = cv::Mat(2,3,CV_32F,cv::Scalar(0.));
        float coef = rand_coef;
        float ang = (random.getRand()-0.5)*coef;
        float c = cos(ang); float s = sin(ang);
        T.at<float>(0,0) = c + (random.getRand()-0.5)*coef;
        T.at<float>(1,1) = c + (random.getRand()-0.5)*coef;
        T.at<float>(0,1) = -s + (random.getRand()-0.5)*coef;
        T.at<float>(1,0) = s + (random.getRand()-0.5)*coef;
        T.at<float>(0,2) = (w/2) - (T.at<float>(0,0)*(w/2) + T.at<float>(0,1)*(h/2));
        T.at<float>(1,2) = (h/2) - (T.at<float>(1,0)*(w/2) + T.at<float>(1,1)*(h/2));
        cv::Mat dst;
        cv::warpAffine(a, dst, T, cv::Size(w,h), cv::INTER_LINEAR, cv::BORDER_REFLECT);
        return dst;
    }
}
