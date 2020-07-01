// =====================================================================================
// 
//       Filename:  main.cpp
// 
//        Version:  1.0
//        Created:  08/05/2020 15:48:58
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Yu Fang (Robotics), yu.fang@iim.ltd
//        Company:  IIM
// 
//    Description:  main function for face detector
// 
// =====================================================================================

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "face_detector.h"

int main(int argc, char** argv)
{
    std::string img_path;
    if (argc == 1) {
        img_path = "../sample.jpg";
    } else if (argc == 2) {
        img_path = argv[1];
    }

    std::string model_path = "/home/fangyu/git/Face-Detector-1MB-with-landmark/converted_models/mobilenet/mobilenetv2.pb";
    const int max_side = 320;
    float threshold = 0.8;
    int num_of_thread = 1;

    Detector* detector = new Detector(model_path);
    detector->setParams(threshold, num_of_thread);
    Timer timer;

    for (int i = 0; i < 10; i++) {
        cv::Mat img = cv::imread(img_path.c_str());
        if (img.empty()) {
            std::cout << "cv imread failed: " << img_path.c_str() << std::endl;
            return -1;
        }

        // scale
        float long_side = std::max(img.cols, img.rows);
        float scale = max_side/long_side;
        std::cout << "scale: " << scale << std::endl;

        cv::Mat img_scale;
        cv::Size size = cv::Size(img.cols*scale, img.rows*scale);
        cv::resize(img, img_scale, size);

        std::vector<bbox> boxes;
        
        timer.tic();

        detector->detect(img_scale, boxes);
        timer.toc("-----total time:");
        
        // draw image
        for (int j = 0; j < boxes.size(); ++j) {
            cv::Rect rect(boxes[j].x1/scale, boxes[j].y1/scale, boxes[j].x2/scale - boxes[j].x1/scale, boxes[j].y2/scale - boxes[j].y1/scale);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
            char test[80];
            sprintf(test, "%f", boxes[j].s);

            cv::putText(img, test, cv::Size((boxes[j].x1/scale), boxes[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
            cv::circle(img, cv::Point(boxes[j].points[0]._x / scale, boxes[j].points[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].points[1]._x / scale, boxes[j].points[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(img, cv::Point(boxes[j].points[2]._x / scale, boxes[j].points[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].points[3]._x / scale, boxes[j].points[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(img, cv::Point(boxes[j].points[4]._x / scale, boxes[j].points[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);
        }

        cv::imwrite("test.png", img);
    }

    return 0;
}
