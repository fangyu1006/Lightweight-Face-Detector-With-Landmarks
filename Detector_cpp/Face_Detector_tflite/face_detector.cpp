// =====================================================================================
// 
//       Filename:  face_detector.cpp
// 
//        Version:  1.0
//        Created:  08/05/2020 11:04:24
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Yu Fang (Robotics), yu.fang@iim.ltd
//        Company:  IIM
// 
//    Description:  face detector using RFB
// 
// ====================================================================================

#include "tensorflow/lite/kernels/register.h"
#include "face_detector.h"

Detector::Detector(const std::string &model_path):
    nms_(0.4),
    threshold_(0.6),
    mean_val_{104.f, 117.f, 123.f},
    num_of_threads_(1)
{
    //input_geometry_ = cv::Size(320, 320);
    //cv::Scalar channel_mean = cv::Scalar(mean_val_[0], mean_val_[1], mean_val_[2]);
    //mean_ = cv::Mat(input_geometry_, CV_32FC3, channel_mean);

    load_graph_tflite(model_path);
}

Detector::~Detector()
{
    std::cout << "destroy detector" << std::endl;
}

void Detector::detect(cv::Mat &img, std::vector<bbox> &boxes)
{
    boxes.clear();
    Timer timer;
    timer.tic();

    int w = img.cols;
    int h = img.rows;

    std::cout << "w, h:  " << w << ", " << h << std::endl;

    input_geometry_ = cv::Size(w, h);
    cv::Scalar channel_mean = cv::Scalar(mean_val_[0], mean_val_[1], mean_val_[2]);
    mean_ = cv::Mat(input_geometry_, CV_32FC3, channel_mean);

    // preprocess
    cv::Mat3f convertedImg;
    //cv::cvtColor(img, convertedImg, cv::COLOR_BGR2RGB);
    img.convertTo(convertedImg, CV_32FC3);
    cv::subtract(convertedImg, mean_, convertedImg);
    // std::cout << convertedImg << std::endl;

    // get input data
    std::cout << "input id: " << input_id << std::endl;
    int input_size = w*h*convertedImg.channels();

//    if (w != 320 || h != 240) {
        interpreter_->ResizeInputTensor(input_id, {1,h,w,3});
        if (interpreter_->AllocateTensors()!=kTfLiteOk){
            std::cout<< "Failed to allocate tensors!"<<std::endl;
        }
  //  }


    
    for (size_t i = 0; i < input_size; i++) {
        //input_data[i] = *((float*)convertedImg.data+i);
        interpreter_->typed_tensor<float>(input_id)[i] = float(*((float*)convertedImg.data+i));
    }
    
    // run net
    if (interpreter_->Invoke() != kTfLiteOk) {
        std::cout<< "Invoke error"<<std::endl;
    }

    timer.toc("det:");

    // create anchor
    std::vector<box> anchor;
    timer.tic();
    createAnchorRetinaface(anchor, w, h);
    timer.toc("anchor:");

    std::cout << "dim size: " <<interpreter_->tensor(interpreter_->outputs()[1])->dims->data[1] << std::endl;; 
    // get output
    std::vector<bbox > total_box;
    float* score_data = interpreter_->typed_output_tensor<float>(0);
    float* box_data = interpreter_->typed_output_tensor<float>(1);
    float* landmark_data = interpreter_->typed_output_tensor<float>(2);

    std::cout << "anchor size: " << anchor.size() << std::endl;
    for (int i = 0; i < anchor.size(); ++i) {
        //std::cout << *(score_data) << ", " << *(score_data+1) << std::endl;
        std::vector<float> score = softmax(*(score_data), *(score_data+1));
        //std::cout << score[0] << ",   " << score[1] << std::endl;
        //std::cout << *(box_data) << ", " << *(box_data+1) << ", " << *(box_data+2) << ", " << *(box_data+3) << std::endl;
        if (score[1] > threshold_) {
            box tmp = anchor[i];
            box tmp1;
            bbox result;

            // loc 
            tmp1.cx = tmp.cx + (*box_data) * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(box_data+1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(box_data+2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(box_data+3) * 0.2);

            result.x1 = std::max((tmp1.cx - tmp1.sx/2) * w, 0.f);
            result.y1 = std::max((tmp1.cy - tmp1.sy/2) * h, 0.f);
            result.x2 = std::min((tmp1.cx + tmp1.sx/2) * w, float(w));
            result.y2 = std::min((tmp1.cy + tmp1.sy/2) * h, float(h));

            // conf
            result.s = score[1];

            // landmark
            for (int j = 0; j < 5; ++j) {
                result.points[j]._x = ( tmp.cx + *(landmark_data + (j<<1)) * 0.1 * tmp.sx ) * w;
                result.points[j]._y = ( tmp.cy + *(landmark_data + (j<<1) + 1) * 0.1 * tmp.sy ) * h;
            }

            total_box.push_back(result);
        }

        score_data += 2;
        box_data += 4;
        landmark_data += 10;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, nms_);
    std::cout << total_box.size() << std::endl;

    for (int j = 0; j < total_box.size(); ++j) {
        boxes.push_back(total_box[j]);
    }

}

inline bool Detector::cmp(bbox a, bbox b)
{
    if (a.s > b.s)
        return true;
    return false;
}

inline std::vector<float> Detector::softmax(float a, float b){
    double sum = std::exp(a) + std::exp(b);
    return std::vector<float>{float(std::exp(a)/sum), float(std::exp(b)/sum)};
}

void Detector::createAnchor(std::vector<box> &anchor, int w, int h)
{
    anchor.clear();
    std::vector<std::vector<int>> feature_map(4), min_sizes(4);
    float steps[] = {8, 16, 32, 64};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }

    std::vector<int> minsize1 = {10, 16, 24};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 48};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {64, 96};
    min_sizes[2] = minsize3;
    std::vector<int> minsize4 = {128, 192, 256};
    min_sizes[3] = minsize4;

    for (int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (int l = 0; l < min_size.size(); ++l) {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }
    }

}

void Detector::createAnchorRetinaface(std::vector<box> &anchor, int w, int h)
{
    anchor.clear();
    std::vector<std::vector<int>> feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }

    std::vector<int> minsize1 = {10, 20};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 64};
    min_sizes[1] = minsize2;
     std::vector<int> minsize3 = {128, 256};
    min_sizes[2] = minsize3;

    for (int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (int l = 0; l < min_size.size(); ++l) {
                    float s_kx = min_size[l] * 1.0 / w;
                    float s_ky = min_size[l] * 1.0 / h;
                    float cx = (j + 0.5) * steps[k] / w;
                    float cy = (i + 0.5) * steps[k] / h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }
    }

}

void Detector::nms(std::vector<bbox> &input_boxes, float nms_thresh)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
            * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }

    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i+1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);

            if (ovr >= nms_thresh) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

void Detector::setParams(float threshold, int num_of_threads)
{
    threshold_ = threshold;
    num_of_threads_ = num_of_threads;
}

TfLiteStatus Detector::load_graph_tflite(const std::string &model_path)
{
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);

    interpreter_->SetNumThreads(num_of_threads_);
    input_id = interpreter_->inputs()[0];
    /*  
    if (interpreter_->AllocateTensors()!=kTfLiteOk){
        std::cout<< "Failed to allocate tensors!"<<std::endl;
    }
    */

    return kTfLiteOk;
}
