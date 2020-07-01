/**
 *          @file  face_detector.h
 *
 *         @brief  face detector using RFB
 *
 *       @version  1.0
 *          @date  07/05/2020 22:27:06
 *        @author  Yu Fang (Robotics), yu.fang@iim.ltd
 * 
 * @section Description
 * 
 *       Revision:  none
 *       Compiler:  g++
 *        Company:  IIM
 * 
 * @section Description
 *
 * -->describe more here<--
 *
 */

#ifndef _FACE_DETECTOR_H_
#define _FACE_DETECTOR_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <stack>
#include "tensorflow/lite/model.h"

using namespace std::chrono;

class Timer
{
public:
    std::stack<high_resolution_clock::time_point> tictoc_stack;

    void tic()
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    double toc(std::string msg = "", bool flag = true)
    {
        double diff = duration_cast<milliseconds>(high_resolution_clock::now() - tictoc_stack.top()).count();
        if (msg.size() > 0) {
            if (flag) {
                std::cout << msg.c_str() << " time elasped " << diff << "ms" << std::endl; 
            }
        }

        tictoc_stack.pop();
        return diff;
    }

    void reset()
    {
        tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};


struct Point {
    float _x;
    float _y;
};

struct bbox {
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Point points[5];
};

struct box {
    float cx;
    float cy;
    float sx;
    float sy;
};

class Detector
{
public:
    Detector(const std::string &model_path);
    ~Detector();
    void detect(cv::Mat &img, std::vector<bbox> &boxes);
    void setParams(float threshold, int num_of_threads);
private:
    TfLiteStatus load_graph_tflite(const std::string &model_path);
    static inline bool cmp(bbox a, bbox b);
    static inline std::vector<float> softmax(float a, float b);
    void createAnchor(std::vector<box> &anchor, int w, int h);
    void createAnchorRetinaface(std::vector<box> &anchor, int w, int h);
    void nms(std::vector<bbox> &input_boxes, float nms_thresh);

private:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;

    int input_id;
    //TfLiteTensor* input_tensor_;
    //TfLiteTensor* output_tensor_;

    cv::Size input_geometry_;
    cv::Mat mean_;
    
    float nms_;
    float threshold_;
    float mean_val_[3];
    int num_of_threads_;
};


#endif //!_FACE_DETECTOR_H_

