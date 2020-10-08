//
// Created by uv on 10/6/20.
//

#ifndef HOG_SVM_DATASET_GENERATING_H
#define HOG_SVM_DATASET_GENERATING_H

#include <opencv2/opencv.hpp>

class DatasetGenerating
{
public:

    DatasetGenerating();

    static cv::Mat ShuffleRows(const cv::Mat &matrix);

    static void DataPreprocessing(cv::Mat& train_data,cv::Mat& train_data_label,
                                  int num0,int num1,std::string address);

    static void HogDataset(cv::Mat &train_data,cv::Mat &train_data_label);
};


#endif //HOG_SVM_DATASET_GENERATING_H
