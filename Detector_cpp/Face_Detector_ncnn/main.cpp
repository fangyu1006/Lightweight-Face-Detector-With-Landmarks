#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <boost/filesystem.hpp>

#include "FaceDetector.h"

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

    string imgPath;
    if  (argc = 1)
    {
        imgPath = "../sample.jpg";
    }
    else if (argc = 2)
    {
        imgPath = argv[1];
    }
    string param = "../model/face.param";
    string bin = "../model/face.bin";
    const int max_side = 320;

    // slim or RFB
    Detector detector(param, bin, false);
    // retinaface
    // Detector detector(param, bin, true);
    Timer timer;

    std::vector<std::string> file_names;
    std::string test_path = "/home/fangyu/Videos/face_with_mask/test/";
    std::string save_path = "/home/fangyu/Videos/face_with_mask/test_results/";
    get_image_names(test_path, file_names);

    for	(auto img_name : file_names){

        std::string imgPath = test_path + img_name;
        cv::Mat img = cv::imread(imgPath.c_str());

        // scale
        float long_side = std::max(img.cols, img.rows);
        float scale = max_side/long_side;
        cv::Mat img_scale;
        cv::Size size = cv::Size(img.cols*scale, img.rows*scale);
        cv::resize(img, img_scale, cv::Size(img.cols*scale, img.rows*scale));

        if (img.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imgPath.c_str());
            return -1;
        }
        std::vector<bbox> boxes;

        timer.tic();

        detector.Detect(img_scale, boxes);
        timer.toc("----total timer:");

        // draw image
        for (int j = 0; j < boxes.size(); ++j) {
            cv::Rect rect(boxes[j].x1/scale, boxes[j].y1/scale, boxes[j].x2/scale - boxes[j].x1/scale, boxes[j].y2/scale - boxes[j].y1/scale);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
            char test[80];
            sprintf(test, "%f", boxes[j].s);

            cv::putText(img, test, cv::Size((boxes[j].x1/scale), boxes[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
            cv::circle(img, cv::Point(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(img, cv::Point(boxes[j].point[4]._x / scale, boxes[j].point[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);
        }
        cv::imwrite(save_path + img_name, img);
    }
    return 0;
}

