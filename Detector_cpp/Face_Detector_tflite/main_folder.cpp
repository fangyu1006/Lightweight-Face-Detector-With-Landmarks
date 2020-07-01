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
#include <boost/filesystem.hpp>

#include "face_detector.h"

using namespace std;
using namespace boost::filesystem;

void get_image_names(const string &folder, std::vector<std::string>& file_names)
{
    path directory(folder);
    directory_iterator itr(directory), end_itr;
    string current_file = itr->path().string();

    for (; itr != end_itr; ++itr) {
        if (is_regular_file(itr->path())) {
            string filename = itr->path().filename().string();
            file_names.push_back(filename);
        }
    }
}

int main(int argc, char** argv)
{
    std::string folder_path;
    if (argc == 1) {
        folder_path = "../images";
    } else if (argc == 2) {
        folder_path = argv[1];
    }

    std::string model_path = "/home/fangyu/git/Face-Detector-1MB-with-landmark/converted_models/mobilenet/mobilenetv2.tflite";
    const int max_side = 320;
    float threshold = 0.9;
    int num_of_thread = 1;

    Detector* detector = new Detector(model_path);
    detector->setParams(threshold, num_of_thread);
    Timer timer;
    int cnt = 0;

    std::vector<std::string> file_names;
    get_image_names(folder_path, file_names);

    std::string save_path = folder_path+"_results/";
    boost::filesystem::create_directories(save_path.c_str());

    for (auto img_name : file_names) {
        std::string img_path = folder_path + "/" + img_name;
        cv::Mat img = cv::imread(img_path.c_str());
        if (img.empty()) {
            std::cout << "cv imread failed: " << img_path.c_str() << std::endl;
            return -1;
        }

        // scale
        float long_side = std::max(img.cols, img.rows);
        float scale = max_side/long_side;
        cv::Mat img_scale;
        cv::Size size = cv::Size(img.cols*scale, img.rows*scale);
        cv::resize(img, img_scale, cv::Size(img.cols*scale, img.rows*scale));

        std::vector<bbox> boxes;
        
        timer.tic();

        detector->detect(img_scale, boxes);
        timer.toc("-----total time:");
        
        cnt += boxes.size();
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

        cv::imwrite(save_path + img_name, img);
    }

    std::cout << "========================" << std::endl;
    std::cout << "total faced detected: " << cnt << std::endl;

    return 0;
}
