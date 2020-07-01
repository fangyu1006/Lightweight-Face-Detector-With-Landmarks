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
    //
    sess = load_graph(model_path.c_str(), &graph);
}

Detector::~Detector()
{
    std::cout << "destroy detector" << std::endl;
    TF_Status* s = TF_NewStatus();
    TF_CloseSession(sess,s);
    TF_DeleteSession(sess,s);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(s);
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
    int input_size = w*h*convertedImg.channels();

    /*  tensorflow related */
    TF_Status * s = TF_NewStatus();
    std::vector<TF_Output> input_names;
    std::vector<TF_Tensor*> input_values;
    TF_Operation* input_name = TF_GraphOperationByName(graph, "input0");
    input_names.push_back({input_name, 0});

    const int64_t dim[4] = {1, h, w, 3};
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dim, 4, convertedImg.ptr(),
            sizeof(float)*input_size, dummy_deallocator, nullptr);
    input_values.push_back(input_tensor);

    std::vector<TF_Output> output_names;
    TF_Operation* output_name = TF_GraphOperationByName(graph, "Concat_223");
    output_names.push_back({output_name,0});
    output_name = TF_GraphOperationByName(graph, "Concat_198");
    output_names.push_back({output_name,0});
    output_name = TF_GraphOperationByName(graph, "Concat_248");
    output_names.push_back({output_name,0});

    std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);

    TF_SessionRun(sess,nullptr,input_names.data(),input_values.data(),input_names.size(),
            output_names.data(),output_values.data(),output_names.size(), nullptr,0,nullptr,s);
    assert(TF_GetCode(s) == TF_OK);               

    timer.toc("det:");

    // create anchor
    std::vector<box> anchor;
    timer.tic();
    createAnchorRetinaface(anchor, w, h);
    timer.toc("anchor:");

    // get output
    std::vector<bbox> total_box;
    const float* score_data = (const float *)TF_TensorData(output_values[0]);
    const float* box_data = (const float *)TF_TensorData(output_values[1]);
    const float* landmark_data = (const float *)TF_TensorData(output_values[2]);

    int anchor_size = anchor.size();
    /*  
    std::vector<std::vector<float>> scores_out(2, std::vector<float>(anchor_size));
    std::vector<std::vector<float>> boxes_out(4, std::vector<float>(anchor_size));
    std::vector<std::vector<float>> landmarks_out(10, std::vector<float>(anchor_size));

    for (int i = 0; i < anchor_size; i++) {
        for (int j = 0; j < 2; j++) {
            scores_out[j][i] = score_data[j];
        }
        for (int j = 0; j < 4; j++) {
            boxes_out[j][i] = box_data[j];
        }
        
        for (int j = 0; j < 10; j++) {
            landmarks_out[j][i] = landmark_data[j];
        }
        score_data += 2;
        box_data += 4;
        landmark_data += 10;
    }

    std::vector<std::vector<float>> scores_reshape = matrixReshape(scores_out, anchor_size, 2);
    std::vector<std::vector<float>> boxes_reshape = matrixReshape(boxes_out, anchor_size, 4);
    std::vector<std::vector<float>> landmarks_reshape = matrixReshape(landmarks_out, anchor_size, 10);
    */
    std::cout << "output dim :" << TF_Dim(output_values[0],1) << std::endl;
    std::cout << "anchor size: " << anchor.size() << std::endl;

    for (int i = 0; i < anchor_size; ++i) {
        //std::cout << *(score_data) << ", " << *(score_data+1) << std::endl;
        std::vector<float> score = softmax(*score_data, *(score_data+1));
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
        score_data+=2;
        box_data += 4;
        landmark_data += 10;
        /*  
        score_d += 2*2;
        box_d += 4*4;
        landmark_d += 10*10;
        if (score_d + 1 > anchor_size*2) {
            flag_score += 1;
            score_d = flag_score;
        }
        
        if (box_d + 1 > anchor_size*4) {
            flag_box += 1;
            box_d = flag_box;
        }

        if (landmark_d + 1 > anchor_size*10) {
            flag_landmark += 1;
            landmark_d = flag_landmark;
        }*/

    }
    std::cout << "bbox size before nms: " << total_box.size() << std::endl;
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

std::vector<std::vector<float>> Detector::matrixReshape(std::vector<std::vector<float>>& nums, int r, int c) 
{
    int x = nums.size();
    int y = nums[0].size();
    if (x * y != r * c) {
        return nums;
    }
    
    std::vector<std::vector<float>> result(r, std::vector<float>(c));
    int cnt = 0;
    for (int row = 0; row < x; ++ row) {
        for (int col = 0; col < y; ++ col) {
            result[cnt / c][cnt % c] = nums[row][col];
            cnt ++;
        }
    }
    
    return result;
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

TF_Session * Detector::load_graph(const char *frozen_fname, TF_Graph **p_graph)
{
    TF_Status* s = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    std::vector<char> model_buf;
    if (!loadTFModel(frozen_fname, model_buf))
        return nullptr;

    TF_Buffer graph_def = {model_buf.data(), model_buf.size(), nullptr};
    
    TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptionsSetPrefix(import_opts, "");
    TF_GraphImportGraphDef(graph, &graph_def, import_opts, s);

    if (TF_GetCode(s) != TF_OK) {
        std::cout<<"load graph failed!\n Error: "<<TF_Message(s)<<std::endl;
        return nullptr;
    }

    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, sess_opts, s);
    assert(TF_GetCode(s) == TF_OK);

    TF_DeleteStatus(s);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteImportGraphDefOptions(import_opts);
    *p_graph = graph;
    return session;
}

bool Detector::loadTFModel(const std::string& fname, std::vector<char>& buf)
{
    std::ifstream fs(fname, std::ios::binary | std::ios::in);
    if (!fs.good()) {
        std::cout << fname << " does not exist" << std::endl;
        return false;
    }
    
    fs.seekg(0, std::ios::end);
    int fsize = fs.tellg();
    fs.seekg(0, std::ios::beg);
    buf.resize(fsize);
    fs.read(buf.data(), fsize);
    fs.close();
    return true;
} 
