//
// Created by uv on 10/6/20.
//

#ifndef HOG_SVM_OBJECT_DETECTION_H
#define HOG_SVM_OBJECT_DETECTION_H

#include <opencv2/opencv.hpp>

class ObjectDetection
{
public:

    ObjectDetection();

    void HogDescriptor();

    cv::Mat HumanDetection(const cv::Mat &origin_image);

private:

    cv::HOGDescriptor *hog_des;
};

#endif //HOG_SVM_OBJECT_DETECTION_H
