/**
 *          @file  alignment.h
 *
 *         @brief  alignment
 *
 *       @version  1.0
 *          @date  19/05/2020 10:37:52
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

#ifndef _ALIGNMENT_H_
#define _ALIGNMENT_H_

#include "face_detector.h"

class Alignment
{
public:
    Alignment();
    ~Alignment(){}
    std::vector<cv::Mat> alignFace(cv::Mat const& img, std::vector<bbox> const& boxes);

private:
    cv::Mat alignOneFace(cv::Mat const& img, bbox const& box);
};


#endif //!_ALIGNMENT_H_

