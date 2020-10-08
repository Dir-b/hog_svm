//
// Created by uv on 10/6/20.
//

#include "dataset_generating.h"


DatasetGenerating::DatasetGenerating()
{

}

cv::Mat DatasetGenerating::ShuffleRows(const cv::Mat &matrix)
{
    std::vector<int> seeds;
    for(int count = 0;count < matrix.rows;++count)
    {
        seeds.push_back(count);
    }

    cv::randShuffle(seeds,7);
    cv::Mat output;
    for(int count = 0;count < matrix.rows;++count)
    {
        output.push_back(matrix.row(seeds[count]));
    }

    return output;
}

void DatasetGenerating::DataPreprocessing(cv::Mat &train_data, cv::Mat &train_data_label,
                                          int num0, int num1,std::string address)
{
    for(int i = 0 ;i < 2;++i)
    {
        int num = 0;
        if(i == 0)
        {
            num = num0;
            address = "/home/uv/auto_pilot/vision/detect_person/data/0/";
//            std::string temp_address = address;
//            temp_address.append("/negative/");
//            address = temp_address;
        }
        if(i == 1)
        {
            num = num1;
            address = "/home/uv/auto_pilot/vision/detect_person/data/1/";
//            std::string temp_address = address;
//            temp_address.append("/positive/");
//            address = temp_address;
        }

        for(int j = 0;j < num;++j)
        {
            std::string temp_address = address;
            temp_address.append(std::to_string(j));
            if(i == 0) temp_address.append(".bmp");
            if(i == 1) temp_address.append(".bmp");

            cv::Mat raw_image = cv::imread(temp_address,1);
            cv::Mat size_image,gray_image;
            cv::resize(raw_image,size_image,cv::Size(64,128));
            cv::cvtColor(size_image,gray_image,cv::COLOR_BGR2GRAY);
            cv::medianBlur(gray_image,gray_image,3);

            cv::HOGDescriptor hog = cv::HOGDescriptor(cv::Size(64,128),
                          cv::Size(16,16),cv::Size(8,8),
                           cv::Size(8,8),9);

            std::vector<float> temp_hog;
            hog.compute(gray_image,temp_hog,cv::Size(1,1),cv::Size(0,0));
            cv::Mat temp_hog1(1,temp_hog.size(),CV_32FC1,temp_hog.data());
            train_data.push_back(temp_hog1);
            train_data_label.push_back(float(i));
        }
    }
}

void DatasetGenerating::HogDataset(cv::Mat &train_data, cv::Mat &train_data_label)
{
    std::cout<<"train data num: "<<train_data.size<<std::endl;
    std::cout<<"train data label num: "<<train_data_label.size<<std::endl;

    int hog_num = 3780; //每张64*128图像的Hog特征数量 32*7*15
    cv::hconcat(train_data,train_data_label,train_data);
    train_data = ShuffleRows(train_data);
    train_data_label = train_data.rowRange(0,train_data.rows).colRange(hog_num,hog_num+1);
    train_data_label.convertTo(train_data_label,4);
    train_data = train_data.colRange(0,hog_num);
}

