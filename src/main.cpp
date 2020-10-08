#include <iostream>
#include "dataset_generating.h"
#include "data_training.h"
#include "object_detection.h"

int main()
{
    cv::Mat train_data,train_data_label;
    std::string address = "/home/uv/auto_pilot/vision/detect_person/data";
    std::string test_address = "/home/uv/auto_pilot/vision/detect_person/data/test/";

    DatasetGenerating::DataPreprocessing(train_data,train_data_label,1016,1427,address);
    DatasetGenerating::HogDataset(train_data,train_data_label);

    DataTraining data_training;
    data_training.SvmTraining(train_data,train_data_label);
    data_training.SvmDetector();

    ObjectDetection object_detection;
    //object_detection.HogDescriptor();

    std::vector<std::string> test_images;
    cv::glob(test_address,test_images);
    std::cout<<"test image num: "<<test_images.size()<<std::endl;

    for(int i= 0;i<test_images.size();++i)
    {
        cv::Mat dst;
        std::string test_image_address = test_images[i];
        cv::Mat test_image = cv::imread(test_image_address,1);

        dst = object_detection.HumanDetection(test_image);
        //cv::imshow("origin",test_image);
        cv::imshow("result",dst);
        cv::waitKey(10);
    }
    return 0;
}
