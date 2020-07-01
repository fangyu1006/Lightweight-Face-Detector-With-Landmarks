// =====================================================================================
// 
//       Filename:  alignment.cpp
// 
//        Version:  1.0
//        Created:  19/05/2020 10:59:28
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Yu Fang (Robotics), yu.fang@iim.ltd
//        Company:  IIM
// 
//    Description:  alignment
// 
// =====================================================================================

Alignment::Alignment()
{

}

cv::Mat Alignment::alignOneFace(cv::Mat const& img, bbox const& box)
{
    cv::Mat warp_mat,warp_dst;
    double mShrinkSz = 1.f;

    warp_mat = similarity_matrix(srcTri,mRotationCoeffCalib,mTranslate);
    mShrinkSz = mShrink;
}


cv::Mat1f Alignment::calcMatU(std::vector<cv::Point2f> const&  dstTri_s)
{
    int num_point = dstTri_s.size();
    cv::Mat1f X(num_point * 2, 4);
   for (int i = 0, p = 0; i < num_point; i++, p+=2) {
       X(p, 0) = dstTri_s[i].x;
       X(p, 1) = dstTri_s[i].y;
       X(p, 2) = 1.f;
       X(p, 3) = 0.f;

       X(p + 1, 0) = dstTri_s[i].y;
       X(p + 1, 1) = -dstTri_s[i].x;
       X(p + 1, 2) = 0.f;
       X(p + 1, 3) = 1.f;
   }

   cv::Mat1f X_t = X.t();
   cv::Mat1f XX = X_t * X;
   return XX.inv() * X_t;
}
