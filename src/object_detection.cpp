//
// Created by uv on 10/6/20.
//

#include <fstream>
#include "object_detection.h"

ObjectDetection::ObjectDetection()
{
    hog_des = new cv::HOGDescriptor();
}

void ObjectDetection::HogDescriptor()
{
    std::vector<float> detector;
    std::ifstream file_in("/home/uv/auto_pilot/vision/hog_svm/result/svm_result.txt",std::ios::in);
    float val = 0.0f;
    while (!file_in.eof())
    {
        file_in >> val;
        detector.push_back(val);
    }
    file_in.close();
    std::cout << "read svm_result.txt" << std::endl;

    this->hog_des->setSVMDetector(detector);
    //this->hog_des->getDefaultPeopleDetector();
}

cv::Mat ObjectDetection::HumanDetection(const cv::Mat &origin_image)
{
    if(origin_image.data == NULL)
    {
        std::cout<<"origin image is null!"<<std::endl;
        return cv::Mat();
    }

    cv::Mat image_detection = origin_image;
    std::vector<cv::Rect> object,object_filtered;

    this->hog_des->detectMultiScale(image_detection,object,0,cv::Size(8,8),
                                   cv::Size(32,32),1.05,2, false);

    for(int i = 0;i<object.size();++i)
    {
        std::cout<<"object num: "<<object.size()<<std::endl;
        cv::Rect rect = object[i];
        for(int j = 0;j<object.size();++j)
        {
            if(j != i && (rect & object[j]) == rect)
            {
                break;
            }
            if(j == object.size())
            {
                object_filtered.push_back(rect);
            }
        }
    }

    for(int i = 0;i<object_filtered.size();++i)
    {
        std::cout<<"object_filtered num: "<<object_filtered.size()<<std::endl;
        cv::Rect rect = object_filtered[i];
        rect.x += cvRound(rect.width * 0.1);
        rect.y += cvRound(rect.height * 0.07);
        rect.width = cvRound(rect.width * 0.8);
        rect.height = cvRound(rect.height * 0.8);

        cv::rectangle(image_detection,rect.tl(),rect.br(),cv::Scalar(0,255,0),1);
    }
    return image_detection;
}


