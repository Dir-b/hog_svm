//
// Created by uv on 10/6/20.
//

#ifndef HOG_SVM_DATA_TRAINING_H
#define HOG_SVM_DATA_TRAINING_H

#include <opencv2/opencv.hpp>

class DataTraining
{
public:

    DataTraining();

    void SvmTraining(cv::Mat &train_data,cv::Mat &train_data_label);

    void SvmDetector();

private:

    cv::Ptr<cv::ml::SVM> svm;

};

#endif //HOG_SVM_DATA_TRAINING_H
